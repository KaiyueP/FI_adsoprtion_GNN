"""Optional force generators. Import MLIP dependencies only during preprocessing."""

def orb_v3(device='cpu', weights_path=None):
    """Orb-v3 direct OMat force model via the pinned orb-models 0.5.5 API.

    The FIRE-GNN paper names Orb-v3 but does not specify this exact variant.
    Disable compilation to avoid substantial cold-start cost for small subsets.
    """
    from pathlib import Path
    from importlib.metadata import version
    if version('orb-models') != '0.5.5':
        raise RuntimeError('Install orb-models==0.5.5 in fire_gnn using FIRE/requirements.txt and a matching torch_scatter build.')
    import os
    import tempfile
    from cached_path import set_cache_dir
    # Home/shared filesystems may reject flock with Errno 524 on NERSC.
    # A process-local /tmp cache supports locking and avoids cross-job races.
    cache = os.environ.get('ORB_CACHE_DIR')
    if cache is None:
        cache = tempfile.mkdtemp(prefix='fire-orb-', dir='/tmp')
    Path(cache).mkdir(parents=True, exist_ok=True)
    set_cache_dir(Path(cache))
    print(f'Orb checkpoint cache: {cache}', flush=True)
    weights_path = weights_path or os.environ.get('ORB_WEIGHTS_PATH')
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    options = dict(device=device, precision='float32-highest', compile=False)
    if weights_path is not None:
        options['weights_path'] = weights_path
    model = pretrained.orb_v3_direct_inf_omat(**options)
    calculator = ORBCalculator(model, device=device)
    if weights_path is not None:
        import hashlib
        path = Path(weights_path).expanduser()
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        calculator.fire_force_source = f'orb-models-0.5.5/{path.name}/sha256-{digest.hexdigest()}'
    else:
        calculator.fire_force_source = 'orb-models-0.5.5/orb-v3-direct-inf-omat-20250404'
    return calculator


def uma(device='cpu', model_name='uma-s-1p2p1', task_name='oc20', inference_settings='batch', cache_dir=None):
    """Optional UMA-S initial forces; import FAIRChem only inside gnn_uma."""
    if task_name != 'oc20':
        raise ValueError('The adsorption UMA adapter requires task_name="oc20".')
    from importlib.metadata import version
    if cache_dir is not None:
        # FAIRChem 2.25 resolves named checkpoints using this import-time setting,
        # independently of get_predict_unit(cache_dir=...). Home may reject flock.
        import os
        from pathlib import Path
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ['FAIRCHEM_CACHE_DIR'] = str(cache_dir)
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as error:
        raise ImportError('UMA requires fairchem-core in the separate gnn_uma environment.') from error
    options = dict(device=device, inference_settings=inference_settings)
    if cache_dir is not None:
        options['cache_dir'] = str(cache_dir)
    predictor = pretrained_mlip.get_predict_unit(model_name, **options)
    calculator = FAIRChemCalculator(predictor, task_name=task_name)
    calculator.fire_force_source = (f'fairchem-core-{version("fairchem-core")}/'
                                   f'{model_name}/{task_name}/{inference_settings}')
    calculator.fire_force_backend = 'uma'
    calculator.fire_force_metadata = dict(model=model_name, task=task_name,
                                          inference_settings=inference_settings,
                                          fairchem_version=version('fairchem-core'))
    return calculator


def uma_atoms_from_record(record):
    """Read ordered pymatgen-style JSON with ASE, without importing pymatgen.

    Preserve site order, lattice, Cartesian positions and partial periodicity.
    Reference labels and relaxed geometry are never passed to UMA.
    """
    import numpy as np
    from ase import Atoms
    material = record['material']
    cell = material['lattice']['matrix']
    symbols, positions = [], []
    for site in material['sites']:
        species = site['species']
        if len(species) != 1 or species[0].get('occu', 1) != 1:
            raise ValueError(f"{record['id']}: UMA preprocessing requires ordered sites.")
        symbols.append(species[0]['element'])
        positions.append(site['xyz'] if 'xyz' in site else np.asarray(site['abc']) @ np.asarray(cell))
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell,
                  pbc=material['lattice'].get('pbc', [True, True, True]))
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError(f"{record['id']}: nonfinite geometry.")
    return atoms


def create_calculator(name='orb', device='cpu', **kwargs):
    """Lazy optional adapters; training does not import either MLIP package."""
    if name == 'orb':
        return orb_v3(device=device, **kwargs)
    if name == 'uma':
        return uma(device=device, **kwargs)
    raise ValueError(f'Unknown MLIP {name!r}; supported built-in choices are orb and uma.')
