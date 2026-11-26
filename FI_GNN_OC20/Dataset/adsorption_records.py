"""Dependency-free validation for the adsorption dataset contract."""
import json
import math


FORCE_MODES = ('none', 'adsorbate', 'adsorbate_nearby', 'all')


def force_mode(config):
    """Explicit mode wins; old checkpoints retain their original force behavior."""
    mode = config.get('force_mode')
    if mode is None:
        mode = 'all' if config.get('use_mlip_forces', False) else 'none'
    if mode not in FORCE_MODES:
        raise ValueError(f'force_mode must be one of {FORCE_MODES}; got {mode!r}')
    return mode


def normalize_force_config(config):
    """Return a consistent checkpoint config without modifying the caller's dict."""
    result = dict(config)
    result['force_mode'] = force_mode(config)
    result['use_mlip_forces'] = result['force_mode'] != 'none'
    if not result['use_mlip_forces']:
        result.pop('force_source', None)
    return result


def validate_forces(record, required=False):
    """Forces are initial-geometry Cartesian vectors in eV/Angstrom, in site order."""
    forces = record.get('mlip_forces')
    if forces is None:
        if required:
            raise ValueError(f"{record['id']}: missing mlip_forces; preprocess all splits first.")
        return
    if (not isinstance(forces, list) or len(forces) != len(record['material']['sites'])
            or any(not isinstance(row, list) or len(row) != 3
                   or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not math.isfinite(v) for v in row) for row in forces)):
        raise ValueError(f"{record['id']}: mlip_forces must be finite N x 3 Cartesian vectors.")
    metadata = record.get('mlip_force_metadata', {})
    if (not isinstance(metadata, dict) or metadata.get('units') != 'eV/Angstrom'
            or metadata.get('geometry') != 'initial'
            or not isinstance(metadata.get('source'), str) or not metadata['source'].strip()):
        raise ValueError(f"{record['id']}: provide initial geometry, eV/Angstrom units and force source metadata.")


def load_records(path, labeled=True):
    with open(path) as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError('Dataset must be a nonempty JSON list.')
    seen = set()
    for record in records:
        identifier = record.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Each record needs a unique, nonempty string id.')
        seen.add(identifier)
        sites = record.get('material', {}).get('sites', [])
        mask = record.get('adsorbate_mask', [])
        if (not sites or len(mask) != len(sites)
                or any(type(v) not in (int, bool) or v not in (0, 1) for v in mask)
                or not any(mask) or all(mask)):
            raise ValueError(f'{identifier}: adsorbate_mask must mark both slab and adsorbate atoms and match sites.')
        validate_forces(record)
        if labeled:
            value = record.get('adsorption_energy')
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{identifier}: adsorption_energy must be a finite number in eV.')
        group = record.get('group_id')
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError(f'{identifier}: group_id must be a nonempty string.')
    return records


def check_splits(*splits):
    """Reject repeated IDs and, when supplied, repeated material groups."""
    seen_ids, seen_groups = set(), set()
    all_records = [record for split in splits for record in split]
    if any('group_id' in r for r in all_records) and not all('group_id' in r for r in all_records):
        raise ValueError('Provide group_id on every record or on none.')
    for records in splits:
        ids = {r['id'] for r in records}
        groups = {r['group_id'] for r in records if 'group_id' in r}
        if seen_ids & ids or seen_groups & groups:
            raise ValueError('Dataset splits overlap in sample IDs or group IDs.')
        seen_ids.update(ids)
        seen_groups.update(groups)
