"""Add initial-geometry forces using Orb, UMA, or a custom ASE calculator.

Run on an allocated compute node. No geometry relaxation or target energies are
passed to the calculator. The factory must return an ASE Calculator.
"""
import argparse
import importlib
import json
from pathlib import Path

from Dataset.adsorption_records import load_records, validate_forces


def add_forces(record, calculator, source, slab_vacuum=None):
    import numpy as np
    if getattr(calculator, 'fire_force_backend', None) == 'uma':
        from mlip_calculators import uma_atoms_from_record
        atoms = uma_atoms_from_record(record)
    else:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        # Original Orb/custom-calculator geometry path remains unchanged.
        structure = Structure.from_dict(record['material'])
        atoms = AseAtomsAdaptor.get_atoms(structure)
        # The legacy adaptor discards partial periodicity; restore it explicitly.
        atoms.set_pbc(structure.lattice.pbc)
    atoms.set_constraint([])
    # Carry the adsorbate identifier through ASE's standard tags (slab=0,
    # adsorbate=2); factories requiring surface/subsurface tags need richer data.
    atoms.set_tags([2 if flag else 0 for flag in record['adsorbate_mask']])
    if slab_vacuum is not None:
        if tuple(atoms.pbc) != (True, True, False):
            raise ValueError('Slab conversion requires PBC=(True,True,False).')
        # Orthogonalize only the nonperiodic cell vector. Preserve all relative
        # initial positions and the two periodic lattice vectors, then translate.
        normal = np.cross(atoms.cell[0], atoms.cell[1])
        normal /= np.linalg.norm(normal)
        projected = atoms.positions @ normal
        height = float(np.ptp(projected)) + slab_vacuum
        cell = atoms.cell.array.copy()
        cell[2] = normal * height
        atoms.set_cell(cell, scale_atoms=False)
        atoms.positions += (slab_vacuum / 2 - projected.min()) * normal
        atoms.set_pbc(True)
    atoms.calc = calculator
    forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError(f"{record['id']}: calculator returned invalid forces.")
    result = dict(record)
    result['mlip_forces'] = forces.tolist()
    result['mlip_force_metadata'] = {
        'units': 'eV/Angstrom', 'geometry': 'initial', 'source': source}
    result['mlip_force_metadata'].update(getattr(calculator, 'fire_force_metadata', {}))
    if slab_vacuum is not None:
        result['mlip_force_metadata'].update(slab_vacuum_A=slab_vacuum,
                                             calculator_pbc=[True, True, True],
                                             calculator_cell_A=atoms.cell.array.tolist())
    validate_forces(result, required=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    calculators = parser.add_mutually_exclusive_group(required=True)
    calculators.add_argument('--mlip', choices=['orb', 'uma'],
                             help='Built-in force generator; original atomic positions are unchanged')
    calculators.add_argument('--calculator-factory',
                             help='Custom importable module:function returning an ASE calculator')
    parser.add_argument('--device', default='cpu', help='Device for a built-in MLIP')
    parser.add_argument('--calculator-kwargs', default='{}', help='JSON keyword arguments for factory')
    parser.add_argument('--source', help='Override provenance; required with a custom calculator factory')
    parser.add_argument('--slab-vacuum', type=float,
                        help='Convert xy-periodic slabs to full PBC with this vacuum gap (Angstrom)')
    args = parser.parse_args()
    if args.slab_vacuum is not None:
        import math
        if not math.isfinite(args.slab_vacuum) or args.slab_vacuum < 13:
            parser.error('--slab-vacuum must be at least 13 Angstrom for Orb-v3 (6 Angstrom cutoff)')
    if args.output.exists():
        parser.error('Output exists; choose a new path. Input files are never overwritten.')
    if args.source is not None and not args.source.strip():
        parser.error('--source must be nonempty')
    if args.calculator_factory and args.source is None:
        parser.error('--source is required with --calculator-factory')
    kwargs = json.loads(args.calculator_kwargs)
    if not isinstance(kwargs, dict):
        parser.error('--calculator-kwargs must be a JSON object')
    records = load_records(args.input, labeled=False)
    if args.mlip:
        from mlip_calculators import create_calculator
        if 'device' in kwargs:
            parser.error('Use --device rather than putting device in --calculator-kwargs for built-in MLIPs')
        calculator = create_calculator(args.mlip, device=args.device, **kwargs)
        source = args.source or calculator.fire_force_source
    else:
        module, name = args.calculator_factory.split(':', 1)
        calculator = getattr(importlib.import_module(module), name)(**kwargs)
        source = args.source
    # Exclusive creation protects original data. Stream each completed record to
    # disk; after interruption the partial JSON is deliberately not trainable.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write('[\n')
        for i, record in enumerate(records):
            enriched = add_forces(record, calculator, source, args.slab_vacuum)
            if i:
                handle.write(',\n')
            json.dump(enriched, handle, allow_nan=False)
            handle.flush()
            print(f"Forces {i + 1}/{len(records)}: {record['id']}", flush=True)
        handle.write('\n]\n')
    print(f'Saved force-enriched records: {args.output}')


if __name__ == '__main__':
    main()
