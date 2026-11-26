"""Export a bounded subset of an official OC20 PyG LMDB to adsorption JSON."""
import argparse
import json
import pickle
from pathlib import Path


def convert_sample(sample, target):
    from pymatgen.core import Lattice, Structure
    from torch_geometric.data import Data

    # Original OC20 releases serialized pre-PyG-2 Data objects.
    if '_store' not in sample.__dict__:
        sample = Data(**{k: v for k, v in sample.__dict__.items() if not k.startswith('_')})

    def tensor(value):
        import torch
        return torch.as_tensor(value).detach().cpu()

    sid = str(sample.sid) if isinstance(sample.sid, str) else str(tensor(sample.sid).item())
    fid = getattr(sample, 'fid', None)
    identifier = sid if fid is None else f'{sid}:{tensor(fid).item()}'
    tags = tensor(sample.tags).reshape(-1)
    if not set(tags.tolist()) <= {0, 1, 2}:
        raise ValueError(f'{identifier}: unexpected OC20 atom tags.')
    # OC20: 0=subsurface, 1=surface, 2=adsorbate.
    mask = (tags == 2).int().tolist()
    cell = tensor(sample.cell).reshape(3, 3).tolist()
    structure = Structure(Lattice(cell, pbc=(True, True, False)),
                          tensor(sample.atomic_numbers).int().tolist(),
                          tensor(sample.pos).tolist(), coords_are_cartesian=True)
    record = {'id': identifier, 'group_id': sid, 'material': structure.as_dict(),
              'adsorbate_mask': mask}
    if target != 'none':
        label = getattr(sample, target, None)
        if label is None:
            raise ValueError(f'{identifier}: missing {target}; check task and split.')
        record['adsorption_energy'] = float(tensor(label).item())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='One trusted official OC20 LMDB file')
    parser.add_argument('output', type=Path)
    parser.add_argument('--target', choices=['y_relaxed', 'y', 'none'], required=True,
                        help='IS2RE: y_relaxed; referenced S2EF energy only: y; unlabeled: none')
    parser.add_argument('--referenced-energy', action='store_true',
                        help='Confirm S2EF y already uses OC20 reference energies')
    parser.add_argument('--limit', type=int, required=True, help='Number of samples to export')
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit must be positive')
    if args.target == 'y' and not args.referenced_energy:
        parser.error('S2EF requires --referenced-energy; raw total energies are not adsorption energies')
    import lmdb
    import random
    from Dataset.adsorption_records import load_records
    environment = lmdb.open(str(args.source), subdir=args.source.is_dir(), readonly=True,
                            lock=False, readahead=False, meminit=False)
    try:
        with environment.begin() as transaction:
            length = transaction.get(b'length')
            count = int(pickle.loads(length)) if length else environment.stat()['entries']
            indices = sorted(random.Random(args.seed).sample(range(count), min(args.limit, count)))
            records = []
            for index in indices:
                raw = transaction.get(str(index).encode())
                if raw is None:
                    raise ValueError(f'Missing numeric LMDB key {index}; unsupported layout.')
                records.append(convert_sample(pickle.loads(raw), args.target))
    finally:
        environment.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(records, handle)
    load_records(args.output, labeled=args.target != 'none')
    print(f'Exported {len(records)} records to {args.output}')


if __name__ == '__main__':
    main()
