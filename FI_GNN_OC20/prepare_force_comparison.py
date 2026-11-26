"""Copy and validate the full 1000/200/200 partitions for a matched force ablation."""
import argparse
import json
from pathlib import Path
from Dataset.adsorption_records import load_records, check_splits

REPO = Path(__file__).resolve().parents[1]


def prepare(root, loss="mse"):
    if root.exists():
        raise FileExistsError(f'Choose a new output directory: {root}')
    sources = {'train': REPO / 'adsorption_runs/shared_data/train.json',
               'validation': REPO / 'adsorption_runs/shared_data/validation.json',
               'test': REPO / 'adsorption_runs/shared_data/test.json'}
    records = {name: load_records(path) for name, path in sources.items()}
    for name, count in (('train', 1000), ('validation', 200), ('test', 200)):
        if len(records[name]) != count:
            raise ValueError(f'{name}: expected {count} structures, found {len(records[name])}')
    check_splits(*records.values())
    config = json.loads((REPO / 'adsorption_runs/shared_data/config.json').read_text())
    config.update(loss=loss, use_mlip_forces=False, seed=13)
    config.pop('force_source', None)
    root.mkdir(parents=True)
    for name, values in records.items():
        (root / f'{name}.json').write_text(json.dumps(values))
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    (root / 'selection.json').write_text(json.dumps(
        {name: [r['id'] for r in values] for name, values in records.items()}, indent=2))
    print(f'Prepared unchanged 1000/200/200 partitions at {root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--loss', choices=['mae', 'mse'], default='mse')
    args = parser.parse_args()
    prepare(args.directory, loss=args.loss)
