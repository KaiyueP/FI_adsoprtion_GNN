"""Perform offline MLIP geometry relaxation (V3).

Relax initial OC20 structures using an MLIP (Orb-v3 or UMA) via ASE optimizers.
Preserves original site order, tags, adsorbate mask, and reference adsorption energy label.
Updates the pymatgen Structure dictionary with the relaxed coordinates R*.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Dataset.adsorption_records import load_records


def _atoms_from_record(record):
    """Construct ASE atoms without requiring pymatgen in the MLIP environment."""
    from ase import Atoms

    material = record['material']
    cell = np.asarray(material['lattice']['matrix'], dtype=float)
    symbols, positions = [], []
    for site in material['sites']:
        species = site['species']
        if len(species) != 1 or species[0].get('occu', 1) != 1:
            raise ValueError(f"{record['id']}: relaxation requires ordered sites.")
        symbols.append(species[0]['element'])
        if 'xyz' in site:
            positions.append(site['xyz'])
        else:
            positions.append(np.asarray(site['abc'], dtype=float) @ cell)
    return Atoms(symbols=symbols, positions=positions, cell=cell,
                 pbc=material['lattice'].get('pbc', [True, True, True]))


def _material_with_positions(material, positions):
    """Copy a pymatgen-style dictionary and replace Cartesian/fractional positions."""
    updated = copy.deepcopy(material)
    cell = np.asarray(updated['lattice']['matrix'], dtype=float)
    inverse_cell = np.linalg.inv(cell)
    if len(positions) != len(updated['sites']):
        raise ValueError('Relaxed coordinate count does not match the input structure.')
    for site, position in zip(updated['sites'], np.asarray(positions, dtype=float)):
        site['xyz'] = position.tolist()
        site['abc'] = (position @ inverse_cell).tolist()
    return updated


def relax_structure(record, calculator, optimizer_cls='BFGS', fmax=0.05, max_steps=50,
                    slab_vacuum=15.0, fix_bottom_layers=False):
    """Relax a single record's initial structure using the provided ASE calculator.

    Returns:
        dict: updated record containing:
            - 'material': pymatgen dict with relaxed cartesian coordinates
            - 'relaxation_metadata': convergence status, steps, initial/final max force, elapsed time
    """
    from ase.optimize import BFGS, LBFGS
    from ase.constraints import FixAtoms

    opt_classes = {'BFGS': BFGS, 'LBFGS': LBFGS}
    Optimizer = opt_classes.get(optimizer_cls, BFGS)

    is_uma = getattr(calculator, 'fire_force_backend', None) == 'uma'
    atoms = _atoms_from_record(record)

    # Carry tags: 2 for adsorbate, 0 for slab
    mask = record['adsorbate_mask']
    atoms.set_tags([2 if m else 0 for m in mask])

    # For Orb-v3 which requires 3D PBC with vacuum
    vacuum_applied = False
    normal_vector = None
    shift_offset = 0.0
    if slab_vacuum is not None and not is_uma:
        if tuple(atoms.pbc) == (True, True, False):
            normal = np.cross(atoms.cell[0], atoms.cell[1])
            normal /= np.linalg.norm(normal)
            projected = atoms.positions @ normal
            height = float(np.ptp(projected)) + slab_vacuum
            cell = atoms.cell.array.copy()
            cell[2] = normal * height
            shift = (slab_vacuum / 2.0 - projected.min()) * normal
            atoms.set_cell(cell, scale_atoms=False)
            atoms.positions += shift
            atoms.set_pbc(True)
            vacuum_applied = True
            normal_vector = normal
            shift_offset = shift

    # Optional: Fix deep bottom slab layers if requested (standard OC20 protocol)
    constraints = []
    if fix_bottom_layers:
        slab_indices = [i for i, m in enumerate(mask) if not m]
        if slab_indices:
            z_coords = [atoms.positions[i, 2] for i in slab_indices]
            z_min = min(z_coords)
            z_cutoff = z_min + 0.35 * (max(z_coords) - z_min)
            fixed_idx = [i for i in slab_indices if atoms.positions[i, 2] <= z_cutoff]
            if fixed_idx:
                constraints.append(FixAtoms(indices=fixed_idx))
    atoms.set_constraint(constraints)

    atoms.calc = calculator

    # Record initial forces before relaxation
    init_forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    init_fmax = float(np.max(np.linalg.norm(init_forces, axis=-1)))

    t0 = time.time()
    converged = False
    steps_taken = 0

    if init_fmax <= fmax:
        converged = True
    else:
        # Run local minimization (logfile=None suppresses terminal noise)
        dyn = Optimizer(atoms, logfile=None, maxstep=0.2)
        try:
            dyn.run(fmax=fmax, steps=max_steps)
            converged = bool(dyn.converged())
            steps_taken = int(dyn.nsteps)
        except Exception as e:
            print(f"Warning: optimization exception for {record['id']}: {e}")
            converged = False

    elapsed = time.time() - t0
    final_forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    final_fmax = float(np.max(np.linalg.norm(final_forces, axis=-1)))

    # Revert vacuum shift if it was applied so coordinates remain on the original lattice
    final_pos = atoms.positions.copy()
    if vacuum_applied and normal_vector is not None:
        final_pos -= shift_offset

    out = copy.deepcopy(record)
    out['material'] = _material_with_positions(record['material'], final_pos)
    out['unrelaxed_material'] = record['material']  # Keep reference to initial structure
    out['relaxation_metadata'] = {
        'calculator': getattr(calculator, 'fire_force_source', 'custom'),
        'optimizer': optimizer_cls,
        'fmax_target': float(fmax),
        'max_steps': int(max_steps),
        'steps_taken': int(steps_taken),
        'converged': bool(converged),
        'initial_fmax_eV_A': init_fmax,
        'final_fmax_eV_A': final_fmax,
        'elapsed_seconds': float(elapsed),
        'timestamp': float(time.time())
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Relax OC20 structures using MLIP via ASE (V3 pipeline).")
    parser.add_argument('--input', type=Path, required=True, help="Input dataset JSON (e.g. validation.json)")
    parser.add_argument('--output', type=Path, required=True, help="Output JSON for relaxed structures")
    parser.add_argument('--mlip', choices=['orb', 'uma'], default='orb', help="MLIP calculator to use")
    parser.add_argument('--device', default='cpu', help="cpu or cuda")
    parser.add_argument('--optimizer', choices=['BFGS', 'LBFGS'], default='BFGS')
    parser.add_argument('--fmax', type=float, default=0.05, help="Target force convergence in eV/Angstrom")
    parser.add_argument('--max-steps', type=int, default=50, help="Maximum relaxation steps")
    parser.add_argument('--slab-vacuum', type=float, default=15.0, help="Vacuum padding for Orb calculator")
    parser.add_argument('--limit', type=int, default=None, help="Optional limit for small test/benchmarking")
    parser.add_argument('--cache-dir', type=Path, default=None)
    args = parser.parse_args()

    from mlip_calculators import create_calculator
    print(f"Loading {args.mlip} calculator on {args.device}...")
    calc_kwargs = {}
    if args.cache_dir is not None and args.mlip == 'uma':
        calc_kwargs['cache_dir'] = args.cache_dir
    calc = create_calculator(args.mlip, device=args.device, **calc_kwargs)

    print(f"Loading records from {args.input}...")
    records = load_records(args.input)
    if args.limit:
        records = records[:args.limit]
    total = len(records)
    print(f"Relaxing {total} structures (optimizer={args.optimizer}, fmax={args.fmax}, max_steps={args.max_steps})...")

    results = []
    t_start = time.time()
    for i, r in enumerate(records):
        relaxed = relax_structure(
            r, calc,
            optimizer_cls=args.optimizer,
            fmax=args.fmax,
            max_steps=args.max_steps,
            slab_vacuum=args.slab_vacuum
        )
        meta = relaxed['relaxation_metadata']
        results.append(relaxed)
        print(f"[{i+1}/{total}] {r['id']}: "
              f"fmax {meta['initial_fmax_eV_A']:.2f} -> {meta['final_fmax_eV_A']:.2f} eV/A "
              f"({meta['steps_taken']} steps, converged={meta['converged']}, {meta['elapsed_seconds']:.2f}s)",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    avg_time = total_time / total if total else 0
    conv_count = sum(1 for r in results if r['relaxation_metadata']['converged'])
    print(f"\nDone! Relaxed {total} structures in {total_time:.1f}s (avg {avg_time:.2f}s/structure).")
    print(f"Convergence rate: {conv_count}/{total} ({conv_count/total*100:.1f}%)")
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
