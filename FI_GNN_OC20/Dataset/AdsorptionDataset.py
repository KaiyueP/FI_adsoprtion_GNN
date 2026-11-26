"""Shared structure featurization for adsorption training and prediction."""
from collections import OrderedDict
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile

import torch
from e3nn.o3 import Irreps, spherical_harmonics
from pymatgen.core import Structure
from torch_geometric.data import Data, Dataset
from torch_scatter import scatter
from .adsorption_records import force_mode, validate_forces


class AdsorptionDataset(Dataset):
    def __init__(self, records, atom_init, config, cache_dir=None):
        super().__init__()
        self.records, self.atom_init, self.config = records, atom_init, config
        # A cap is a target count: keep the entire distance shell at its boundary.
        # Absolute tolerance is in Angstrom, independent of atom index or orientation.
        self.neighbor_tie_tolerance = config.get('neighbor_tie_tolerance', 1e-5)
        if (isinstance(self.neighbor_tie_tolerance, bool)
                or not isinstance(self.neighbor_tie_tolerance, (int, float))
                or not math.isfinite(self.neighbor_tie_tolerance)
                or self.neighbor_tie_tolerance <= 0):
            raise ValueError('neighbor_tie_tolerance must be finite and positive (Angstrom).')
        if type(config['max_neighbors']) is not int or config['max_neighbors'] <= 0:
            raise ValueError('max_neighbors must be a positive integer.')
        self.force_mode = force_mode(config)
        self.use_forces = self.force_mode != 'none'
        self.force_radius = config.get('force_radius', 3.0)
        if self.force_mode == 'adsorbate_nearby':
            if (isinstance(self.force_radius, bool) or not isinstance(self.force_radius, (int, float))
                    or not math.isfinite(self.force_radius) or self.force_radius <= 0):
                raise ValueError('force_radius must be finite and positive (Angstrom).')
        self.local_pool_radius = config.get('local_pool_radius', 3.0)
        if config.get('pooling') == 'local_mean':
            if (isinstance(self.local_pool_radius, bool) or not isinstance(self.local_pool_radius, (int, float))
                    or not math.isfinite(self.local_pool_radius) or self.local_pool_radius <= 0):
                raise ValueError('local_pool_radius must be finite and positive (Angstrom).')
        self.force_scale = config.get('force_scale', 1.0)
        if self.use_forces:
            if (isinstance(self.force_scale, bool) or not isinstance(self.force_scale, (int, float))
                    or not math.isfinite(self.force_scale) or self.force_scale <= 0):
                raise ValueError('force_scale must be finite and positive (eV/Angstrom).')
            for record in records:
                validate_forces(record, required=True)
                if (config.get('force_source') is not None
                        and record['mlip_force_metadata']['source'] != config['force_source']):
                    raise ValueError(f"{record['id']}: MLIP source differs from the checkpoint.")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._graph_cache = OrderedDict()
        self._cache_keys = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Bump the version whenever the graph construction changes.
            settings = {'version': 'complete-neighbor-shells-v2', 'atom_init': atom_init,
                        'neighbor_tie_tolerance': self.neighbor_tie_tolerance,
                        'z_symmetry_breaking': config.get('z_symmetry_breaking', False),
                        'use_mlip_forces': self.use_forces, 'force_scale': self.force_scale,
                        'force_mode': self.force_mode,
                        'force_radius': self.force_radius if self.force_mode == 'adsorbate_nearby' else None,
                        **{key: config[key] for key in ('radius', 'max_neighbors', 'lmax_h')}}
            if config.get('pooling') == 'local_mean':
                settings.update(local_pool_mask_version=1, local_pool_radius=self.local_pool_radius)
            self._feature_key = hashlib.sha256(json.dumps(
                settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Mac workers use spawn; do not duplicate the parent's tensor cache.
        state['_graph_cache'] = OrderedDict()
        return state

    def len(self):
        return len(self.records)

    def get(self, idx):
        if self.cache_dir is None:
            return self._build_graph(idx)
        if idx in self._graph_cache:
            graph = self._graph_cache.pop(idx)
        else:
            if idx not in self._cache_keys:
                payload = json.dumps(self.records[idx], sort_keys=True, separators=(',', ':'))
                self._cache_keys[idx] = hashlib.sha256(
                    (self._feature_key + payload).encode()).hexdigest()
            path = self.cache_dir / f'{self._cache_keys[idx]}.pt'
            if path.exists():
                # Cache files contain tensors/scalars, never pickled Data objects.
                graph = Data(**torch.load(path, map_location='cpu', weights_only=True))
            else:
                graph = self._build_graph(idx)
                # Atomic replacement allows two workers to request the same graph.
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    torch.save(graph.to_dict(), temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        self._graph_cache[idx] = graph
        # Bound RAM per process; the complete cache remains on disk.
        if len(self._graph_cache) > 128:
            self._graph_cache.popitem(last=False)
        return graph.clone()

    def _build_graph(self, idx):
        record = self.records[idx]
        structure = Structure.from_dict(record['material'])
        if not structure.is_ordered:
            raise ValueError(f"{record['id']}: disordered structures are unsupported.")
        features = [self.atom_init[str(site.specie.number)] for site in structure]
        x = torch.tensor(features, dtype=torch.float32)
        mask = torch.tensor(record['adsorbate_mask'], dtype=torch.float32).reshape(-1, 1)
        sources, targets, vectors = [], [], []
        for i, neighbors in enumerate(structure.get_all_neighbors(self.config['radius'])):
            neighbors = sorted(neighbors, key=lambda n: n.nn_distance)
            if len(neighbors) > self.config['max_neighbors']:
                boundary = neighbors[self.config['max_neighbors'] - 1].nn_distance
                # A strict slice can pick different members of a crystalline shell
                # after rotation, translation, or atom reordering. Include every
                # tied neighbor; never create edges beyond the radius cutoff.
                neighbors = [n for n in neighbors
                             if n.nn_distance <= boundary + self.neighbor_tie_tolerance]
            for neighbor in neighbors:
                sources.append(i)
                targets.append(neighbor.index)
                vectors.append((neighbor.coords - structure[i].coords).tolist())
        if not vectors:
            raise ValueError(f"{record['id']}: entire structure has no neighbors; inspect geometry/cutoff.")
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        r_ij = torch.tensor(vectors, dtype=torch.float32)
        edge_attr = spherical_harmonics(
            Irreps.spherical_harmonics(self.config['lmax_h']), r_ij[:, [1, 2, 0]],
            normalize=True, normalization='component')
        node_attr = scatter(edge_attr, edge_index[1], dim=0, reduce='mean', dim_size=len(structure))
        incoming = torch.bincount(edge_index[1], minlength=len(structure))
        # Keep isolated atoms in x and in graph pooling. They have no preferred
        # direction: higher-order angular attributes stay zero. With component
        # normalization the l=0 spherical harmonic is 1; retain that scalar so
        # the embedding can still represent the isolated atom's species/mask.
        # Do not create long-range edges or zero-length directional self-edges.
        node_attr[incoming == 0, 0] = 1.0
        inputs = [x, mask]
        force_mask = torch.zeros(len(structure), dtype=torch.bool)
        if self.force_mode == 'all':
            force_mask[:] = True
        elif self.force_mode in ('adsorbate', 'adsorbate_nearby'):
            force_mask = mask.reshape(-1).bool().clone()
            if self.force_mode == 'adsorbate_nearby':
                # Use the full structure and periodic images, independently of
                # the message graph's cutoff and truncated neighbor count.
                for atom_index in torch.where(force_mask)[0].tolist():
                    for neighbor in structure.get_neighbors(structure[atom_index], self.force_radius):
                        force_mask[int(neighbor.index)] = True
        if self.use_forces:
            # Match the [y,z,x] permutation used for spherical harmonics above.
            # One polar-vector irrep (1o); preserve magnitude rather than normalize.
            forces = torch.tensor(record['mlip_forces'], dtype=torch.float32)
            forces = forces.masked_fill(~force_mask.unsqueeze(-1), 0.)
            inputs.append(forces[:, [1, 2, 0]] / self.force_scale)
        if self.config.get('z_symmetry_breaking', False):
            # Fixed Cartesian +z in the [y,z,x] model coordinate frame.
            normal = x.new_tensor([0., 1., 0.]).expand(len(structure), -1)
            inputs.append(normal)
        graph = Data(
            x=torch.cat(inputs, dim=1), edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            edge_dis=r_ij.norm(dim=-1, keepdim=True), r_ij=r_ij,
            pos=torch.tensor(structure.cart_coords, dtype=torch.float32),
            sample_id=record['id'], adsorbate_mask=mask.reshape(-1).bool(),
            force_input_mask=force_mask)
        if self.config.get('pooling') == 'local_mean':
            # Include each unique atom once; full PBC-aware geometry, no edge truncation.
            local_mask = graph.adsorbate_mask.clone()
            for atom_index in torch.where(graph.adsorbate_mask)[0].tolist():
                for neighbor in structure.get_neighbors(structure[atom_index], self.local_pool_radius):
                    local_mask[int(neighbor.index)] = True
            graph.local_pool_mask = local_mask
        if 'adsorption_energy' in record:
            graph.y = torch.tensor([[record['adsorption_energy']]], dtype=torch.float32)
        return graph
