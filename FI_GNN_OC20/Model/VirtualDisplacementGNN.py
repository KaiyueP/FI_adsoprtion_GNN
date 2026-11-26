"""Equivariant Virtual Displacement & Interface Work GNN (V2).

Physics rationale:
1. Forces are gradients, not static features. They point in the direction
   atoms move during relaxation.
2. The virtual displacement layer applies a learned, bounded displacement
   step: delta_R_i = step_size_i * tanh(|F_i| / sigma) * (F_i / |F_i|).
3. The graph relative displacement vectors r_ij' and spherical harmonics
   are updated accordingly before message passing.
4. A physical interface work scalar W = sum_{i in ads} F_i . delta_R_i
   is computed directly and contributes an invariant energy correction.
5. All operations preserve exact E(3) equivariance / invariance:
   - Rotations and reflections of (pos, F, r_ij) transform consistently.
   - Work W is a scalar dot-product invariant.
"""
import math
import torch
from torch import nn
from e3nn.o3 import Irreps, spherical_harmonics
from torch_scatter import scatter

from .SEMPNN import SEMPNN
from .Layers.SEMPLayer import SEMPLayer
from .BalancedIrreps import BalancedIrreps
from .PooledGraphNN import adsorbate_mean_pool
from Dataset.adsorption_records import force_mode


class RadialMessageLayer(SEMPLayer):
    """Scalar invariant gates act equally on every component of each irrep."""
    def __init__(self, *args, radial_width=32, **kwargs):
        super().__init__(*args, **kwargs)
        count = kwargs['additional_message_irreps'].dim
        self.gates = nn.Sequential(
            nn.Linear(count - 1, radial_width),
            nn.SiLU(),
            nn.Linear(radial_width, self.hidden_irreps.num_irreps)
        )
        indices = []
        gate = 0
        for multiplicity, irrep in self.hidden_irreps:
            for _ in range(multiplicity):
                indices.extend([gate] * irrep.dim)
                gate += 1
        self.register_buffer('gate_index', torch.tensor(indices, dtype=torch.long), persistent=False)

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        msg = super().message(x_i, x_j, edge_attr, additional_message_features)
        gates = 2 * torch.sigmoid(self.gates(additional_message_features[:, :-1]))
        envelope = additional_message_features[:, -1:]
        return msg * gates[:, self.gate_index] * envelope / math.sqrt(24.0)


class RadialSEMPNN(SEMPNN):
    def __init__(self, *args, radial_width=32, **kwargs):
        self.radial_width = radial_width
        super().__init__(*args, **kwargs)

    def _make_layer(self):
        return RadialMessageLayer(
            self.hidden_irreps, self.hidden_irreps, self.hidden_irreps,
            self.edge_attr_irreps, self.node_attr_irreps, norm=self.norm,
            additional_message_irreps=self.additional_message_irreps,
            radial_width=self.radial_width
        )


class VirtualDisplacementGNN(nn.Module):
    """V2 Architecture: Virtual coordinate shift + Physical Work readout."""

    def __init__(self, config, feature_count):
        super().__init__()
        if config.get('z_symmetry_breaking', False):
            raise ValueError('virtual_displacement_v2 uses no fixed Cartesian reference vector.')

        self.use_forces = force_mode(config) != 'none'
        self.scalar_count = feature_count - 3 * int(self.use_forces)
        self.radius = float(config['radius'])
        self.lmax_h = config['lmax_h']
        radial_count = config.get('radial_basis_count', 16)
        radial_width = config.get('radial_hidden_features', 32)
        
        # Max physical displacement cap (default 0.35 Angstrom to avoid unphysical warping)
        self.max_displacement = float(config.get('max_virtual_step_A', 0.35))
        self.force_sigma = float(config.get('force_step_sigma', 3.0))

        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('radius must be finite and positive.')

        self.register_buffer('radial_centers', torch.linspace(0, self.radius, radial_count))
        self.radial_width_A = self.radius / (radial_count - 1)

        angular = Irreps.spherical_harmonics(config['lmax_h'])
        hidden = BalancedIrreps(config['lmax_h'], config['hidden_features'])

        common = dict(
            hidden_irreps=hidden,
            output_irreps_lst=[Irreps('1x0e')],
            edge_attr_irreps=angular,
            node_attr_irreps=angular,
            num_graph_layers=config['num_graph_layers'],
            num_prepool_layers=config['num_prepool_layers'],
            num_postpool_layers=config['num_postpool_layers'],
            norm=config['layer_norm'],
            pooling=config['pooling'],
            radial_width=radial_width
        )

        # Baseline geometry branch (evaluates on original R_0)
        self.geometry = RadialSEMPNN(
            input_irreps=Irreps(f'{self.scalar_count}x0e'),
            additional_message_irreps=Irreps(f'{radial_count + 1}x0e'),
            **common
        )

        if self.use_forces:
            # 1. Learned scalar step-size predictor per atom: predicts alpha_i in [0, max_displacement]
            # Inputs: scalar species features + adsorbate mask + log1p(|F_i|)
            self.step_net = nn.Sequential(
                nn.Linear(self.scalar_count + 1, radial_width),
                nn.SiLU(),
                nn.Linear(radial_width, 1),
                nn.Sigmoid()  # outputs in [0, 1]
            )

            # 2. Quasi-relaxed geometry branch (operates on shifted coordinates R')
            # Receives scalar species + updated spherical harmonics + invariant force projections
            self.shifted_gnn = RadialSEMPNN(
                input_irreps=Irreps(f'{self.scalar_count + 1}x0e+1x1o'),
                additional_message_irreps=Irreps(f'{radial_count + 6}x0e'),
                **common
            )

            # Zero-initialize the shifted GNN readout head so training starts from baseline geometry
            final = self.shifted_gnn.mlp.outputs[0][-1]
            for p in final.parameters():
                nn.init.zeros_(p)

            # 3. Direct physical work weight parameter (learns scaling between mechanical work and adsorption delta E)
            self.work_weight = nn.Parameter(torch.zeros(1))

    def compute_virtual_displacements(self, graph, forces):
        """Compute E(3)-equivariant displacements delta_R along force vectors.
        
        Matches model coordinate frame [y, z, x].
        """
        scalars = graph.x[:, :self.scalar_count]
        f_norm = forces.norm(dim=-1, keepdim=True)
        log_f = torch.log1p(f_norm)

        # Predict per-atom step fraction in [0, max_displacement]
        step_fraction = self.step_net(torch.cat([scalars, log_f], dim=-1))
        step_magnitude = step_fraction * self.max_displacement

        # Smooth saturation: tanh(|F| / sigma) avoids huge steps for repulsive force outliers
        saturation = torch.tanh(f_norm / self.force_sigma)
        unit_force = forces / f_norm.clamp_min(1e-8)

        # Equivariant displacement vector
        delta_r = step_magnitude * saturation * unit_force
        
        # Slab atoms with zero input force naturally get delta_r = 0
        return delta_r, f_norm

    def edge_features(self, distances, unit_vectors=None, forces=None):
        """Compute radial Gaussian basis + cosine cutoff envelope + invariants."""
        ratio = (distances / self.radius).clamp(0, 1)
        envelope = 0.5 * (torch.cos(math.pi * ratio) + 1.0)
        envelope = torch.where(distances < self.radius, envelope, torch.zeros_like(envelope))
        radial = torch.exp(-0.5 * ((distances - self.radial_centers) / self.radial_width_A) ** 2)
        radial = radial * envelope

        if forces is None or unit_vectors is None:
            return torch.cat([radial, envelope], dim=-1)

        fi, fj = forces[0], forces[1]
        ni, nj = fi.norm(dim=-1, keepdim=True), fj.norm(dim=-1, keepdim=True)
        cosine = (fi * fj).sum(dim=-1, keepdim=True) / (ni * nj).clamp_min(1e-8)
        
        invariants = torch.cat([
            torch.asinh((fi * unit_vectors).sum(dim=-1, keepdim=True)),
            torch.asinh((fj * unit_vectors).sum(dim=-1, keepdim=True)),
            torch.log1p(ni), torch.log1p(nj),
            cosine.clamp(-1, 1)
        ], dim=-1)

        return torch.cat([radial, invariants, envelope], dim=-1)

    def forward_components(self, graph):
        # 1. Evaluate baseline initial geometry
        geometric = graph.clone()
        geometric.x = graph.x[:, :self.scalar_count]
        geometric.edge_dis = self.edge_features(graph.edge_dis)
        e_geometry = self.geometry(geometric)[0]

        correction = torch.zeros_like(e_geometry)
        work_energy = torch.zeros_like(e_geometry)
        gate = torch.zeros_like(e_geometry)

        if self.use_forces:
            # Forces in [y, z, x] frame
            forces = graph.x[:, self.scalar_count:self.scalar_count + 3]
            source, target = graph.edge_index

            # 2. Virtual step: compute delta_r
            delta_r, f_norm = self.compute_virtual_displacements(graph, forces)

            # Mechanical work done by adsorbate forces: W = sum_{i in ads} F_i . delta_r_i
            # F is in eV/Angstrom, delta_r is in Angstrom -> units of eV
            local_work = (forces * delta_r).sum(dim=-1, keepdim=True)
            graph_work = adsorbate_mean_pool(local_work, graph.batch, graph.adsorbate_mask)
            work_energy = self.work_weight * graph_work

            # 3. Update edge vectors with virtual displacement
            # r_ij' = r_ij + (delta_r_target - delta_r_source)
            # Match coordinate permutation: r_ij in graph is permuted as [y, z, x]
            r_ij_shifted = graph.r_ij[:, [1, 2, 0]] + (delta_r[target] - delta_r[source])
            new_distances = r_ij_shifted.norm(dim=-1, keepdim=True)
            new_units = r_ij_shifted / new_distances.clamp_min(1e-8)

            # Recompute spherical harmonics on shifted geometry
            angular_irreps = Irreps.spherical_harmonics(self.lmax_h)
            new_edge_attr = spherical_harmonics(
                angular_irreps, new_units,
                normalize=True, normalization='component'
            )
            new_node_attr = scatter(
                new_edge_attr, target, dim=0, reduce='mean', dim_size=graph.num_nodes
            )

            # 4. Forward pass on shifted geometry
            shifted_graph = graph.clone()
            shifted_graph.x = torch.cat([geometric.x, torch.log1p(f_norm), forces], dim=-1)
            shifted_graph.edge_attr = new_edge_attr
            shifted_graph.node_attr = new_node_attr
            shifted_graph.edge_dis = self.edge_features(
                new_distances, new_units, (forces[source], forces[target])
            )

            raw_correction = self.shifted_gnn(shifted_graph)[0]

            # Bounded magnitude gate ensures exactly 0 correction if forces are all zero
            mean_f = adsorbate_mean_pool(torch.log1p(f_norm), graph.batch, graph.adsorbate_mask)
            gate = mean_f / (1.0 + mean_f)
            correction = gate * raw_correction + work_energy

        total = e_geometry + correction
        return {
            'geometry': e_geometry,
            'force_correction': correction,
            'work_energy': work_energy,
            'force_gate': gate,
            'total': total
        }

    def forward(self, graph):
        return [self.forward_components(graph)['total']]

