"""Train a scalar adsorption-energy SEGNN or predict with its saved checkpoint."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from e3nn.o3 import Irreps
from torch_geometric.loader import DataLoader

from Dataset.AdsorptionDataset import AdsorptionDataset
from Dataset.adsorption_records import load_records, check_splits, validate_forces, force_mode, normalize_force_config, FORCE_MODES
from Model.SEMPNN import SEMPNN
from Model.BalancedIrreps import BalancedIrreps

ROOT = Path(__file__).resolve().parent


def initialize_worker(worker_id):
    # This function must be at module scope for Mac's spawn worker startup.
    torch.set_num_threads(1)


def graph_loader(dataset, batch_size, shuffle=False, num_workers=0, persistent=False, sampler=None, pin_memory=False, generator=None):
    options = dict(batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                   num_workers=num_workers, sampler=sampler, pin_memory=pin_memory, generator=generator)
    if num_workers:
        options.update(multiprocessing_context='spawn', worker_init_fn=initialize_worker,
                       persistent_workers=persistent, prefetch_factor=2)
    return DataLoader(dataset, **options)


def build_model(config, feature_count):
    architecture = config.get('architecture', 'sempnn')
    if architecture == 'force_residual_v1':
        from Model.ForceResidualGNN import ForceResidualGNN
        return ForceResidualGNN(config, feature_count)
    if architecture == 'virtual_displacement_v2':
        from Model.VirtualDisplacementGNN import VirtualDisplacementGNN
        return VirtualDisplacementGNN(config, feature_count)
    if architecture != 'sempnn':
        raise ValueError(f'Unknown architecture: {architecture}')
    angular = Irreps.spherical_harmonics(config['lmax_h'])
    vector_count = int(force_mode(config) != 'none') + int(config.get('z_symmetry_breaking', False))
    scalar_count = feature_count - 3 * vector_count
    input_irreps = f'{scalar_count}x0e'
    if vector_count:
        input_irreps += f' + {vector_count}x1o'
    return SEMPNN(
        input_irreps=Irreps(input_irreps),
        hidden_irreps=BalancedIrreps(config['lmax_h'], config['hidden_features']),
        output_irreps_lst=[Irreps('1x0e')],
        edge_attr_irreps=angular, node_attr_irreps=angular,
        additional_message_irreps=Irreps('1x0e'),
        num_graph_layers=config['num_graph_layers'],
        num_prepool_layers=config['num_prepool_layers'],
        num_postpool_layers=config['num_postpool_layers'],
        norm=config['layer_norm'], pooling=config['pooling'])


def epoch(model, loader, device, mean, std, optimizer=None, loss_name='mae'):
    model.train(optimizer is not None)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    with torch.set_grad_enabled(optimizer is not None):
        for graph in loader:
            graph = graph.to(device, non_blocking=True)
            prediction = model(graph)[0]
            criterion = (torch.nn.functional.mse_loss if loss_name == 'mse'
                         else torch.nn.functional.l1_loss)
            loss = criterion(prediction, (graph.y - mean) / std)
            finite = torch.isfinite(loss).to(torch.int32)
            if dist.is_initialized() and optimizer is not None:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise ValueError('Nonfinite loss; inspect input data and model settings.')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            error = (prediction.detach() * std + mean - graph.y).double()
            totals[0] += error.abs().sum()
            totals[1] += error.square().sum()
            totals[2] += error.numel()
    if dist.is_initialized():
        dist.all_reduce(totals)
    absolute, squared, count = totals.tolist()
    if not count:
        raise ValueError('Cannot evaluate an empty dataset.')
    return {'mae_eV': absolute / count, 'mse_eV2': squared / count,
            'rmse_eV': (squared / count) ** .5}


def train(args):
    if args.num_workers < 0:
        raise ValueError('num_workers must be nonnegative.')
    if args.patience < 0:
        raise ValueError('patience must be nonnegative (0 disables early stopping).')
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank, world_size = 0, 1
    device = torch.device(args.device)
    if distributed:
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Distributed training requires CUDA; launch with torchrun --device cuda.')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl')
        rank, world_size = dist.get_rank(), dist.get_world_size()
    elif device.type == 'cuda':
        device = torch.device('cuda', device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    resume = bool(getattr(args, 'resume', False))
    if resume and distributed:
        raise ValueError('Checkpoint resume currently supports single-process training only.')
    if resume and device.type != 'cpu':
        raise ValueError('Checkpoint resume currently supports CPU training only.')
    # Fail promptly for a used run name; permit retrying an empty failed run.
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not resume:
        raise FileExistsError(f'Output directory is not empty: {args.output}; use a new run directory.')
    if resume and not (args.output / 'last.pt').is_file():
        raise FileNotFoundError(f'Cannot resume without {args.output / "last.pt"}.')
    if distributed:
        dist.barrier()
    config = json.loads(args.config.read_text())
    if getattr(args, 'pooling', None) is not None:
        config['pooling'] = args.pooling
    if getattr(args, 'adsorbate_pooling', None) is not None:
        config['pooling'] = 'adsorbate_mean' if args.adsorbate_pooling else 'mean'
    if getattr(args, 'z_symmetry_breaking', False):
        config['z_symmetry_breaking'] = True
    if getattr(args, 'force_mode', None) is not None:
        config['force_mode'] = args.force_mode
    elif getattr(args, 'use_mlip_forces', False):
        config['force_mode'] = 'all'
    if getattr(args, 'force_radius', None) is not None:
        config['force_radius'] = args.force_radius
    config = normalize_force_config(config)
    config.setdefault('neighbor_tie_tolerance', 1e-5)
    if getattr(args, 'force_scale', None) is not None:
        config['force_scale'] = args.force_scale
    if getattr(args, 'batch_size', None) is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    for key in ('epochs', 'batch_size', 'max_neighbors', 'num_graph_layers'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if config['radius'] <= 0:
        raise ValueError('radius must be positive.')
    if config.get('loss', 'mae') not in ('mae', 'mse'):
        raise ValueError('config loss must be mae or mse.')
    selection_metric = 'mse_eV2' if config.get('loss') == 'mse' else 'mae_eV'
    torch.manual_seed(config['seed'])
    training, validation = load_records(args.train), load_records(args.validation)
    testing = load_records(args.test) if args.test else []
    check_splits(training, validation, testing)
    if config.get('use_mlip_forces', False):
        sources = set()
        for record in training + validation + testing:
            validate_forces(record, required=True)
            sources.add(record['mlip_force_metadata']['source'])
        if len(sources) != 1:
            raise ValueError('Use the same MLIP model/checkpoint source for all splits.')
        config['force_source'] = sources.pop()
    atom_init = json.loads((ROOT / 'Dataset/atom_init.json').read_text())
    def loader(records, shuffle=False):
        dataset = AdsorptionDataset(records, atom_init, config, cache_dir=args.cache_dir)
        sampler = None
        if distributed:
            # Training pads to equal step counts; evaluation never pads/duplicates.
            sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                          shuffle=True, seed=config['seed']) if shuffle
                       else list(range(rank, len(dataset), world_size)))
        generator = None
        if config.get('independent_loader_rng', False):
            generator = torch.Generator().manual_seed(config['seed'] + (1000 if shuffle else 2000))
        return graph_loader(dataset, config['batch_size'], shuffle=shuffle,
                            num_workers=args.num_workers, sampler=sampler, generator=generator,
                            persistent=args.num_workers > 0, pin_memory=device.type == 'cuda')
    train_loader, val_loader = loader(training, True), loader(validation)
    labels = torch.tensor([r['adsorption_energy'] for r in training], dtype=torch.float64)
    mean, std = labels.mean().item(), labels.std(unbiased=False).item()
    std = std if std > 1e-12 else 1.0
    feature_count = train_loader.dataset[0].x.shape[1]
    model = build_model(config, feature_count).to(device)
    raw_model = model
    if distributed:
        # The adsorption forward path uses all trainable parameters.
        model = DistributedDataParallel(model, device_ids=[device.index],
                                        find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    execution = {'device': str(device), 'torch_threads': torch.get_num_threads(),
                 'architecture': config.get('architecture', 'sempnn'),
                 'parameter_count': sum(p.numel() for p in raw_model.parameters()),
                 'independent_loader_rng': config.get('independent_loader_rng', False),
                 'training_graph_workers': args.num_workers,
                 'evaluation_graph_workers': args.num_workers,
                 'world_size': world_size, 'batch_size_per_gpu': config['batch_size'],
                 'effective_batch_size': config['batch_size'] * world_size,
                 'early_stopping_patience': args.patience,
                 'graph_cache': str(args.cache_dir) if args.cache_dir else None}
    if rank == 0:
        print(json.dumps(execution), flush=True)
    best, history, stale_epochs, start_step = float('inf'), [], 0, 1
    checkpoint_path = args.output / 'best.pt'
    last_checkpoint_path = args.output / 'last.pt'
    if resume:
        last = torch.load(last_checkpoint_path, map_location=device, weights_only=True)
        if last['config'] != config:
            raise ValueError('Resume configuration differs from the saved training configuration.')
        if last['feature_count'] != feature_count:
            raise ValueError('Resume dataset feature count differs from the saved checkpoint.')
        if abs(float(last['target_mean']) - mean) > 1e-12 or abs(float(last['target_std']) - std) > 1e-12:
            raise ValueError('Resume training labels differ from the saved checkpoint statistics.')
        raw_model.load_state_dict(last['model_state'])
        optimizer.load_state_dict(last['optimizer_state'])
        history = last['history']
        best = float(last['best'])
        stale_epochs = int(last['stale_epochs'])
        start_step = int(last['epoch']) + 1
        if not history or int(history[-1]['epoch']) != start_step - 1:
            raise ValueError('Resume checkpoint history is inconsistent with its epoch.')
        torch.set_rng_state(last['torch_rng_state'].cpu())
        print(f'Resuming training after epoch {start_step - 1}; '
              f'best validation {selection_metric}={best:.8g}, stale epochs={stale_epochs}.', flush=True)
    already_stopped = bool(args.patience and stale_epochs >= args.patience)
    if already_stopped:
        print(f'Training checkpoint already satisfies patience={args.patience}; no more epochs needed.', flush=True)
    steps = range(start_step, config['epochs'] + 1) if not already_stopped else ()
    for step in steps:
        started = time.perf_counter()
        if distributed:
            train_loader.sampler.set_epoch(step)
        if rank == 0:
            print(f'Epoch {step}/{config["epochs"]} started (first epoch may build cached graphs).', flush=True)
        train_metrics = epoch(model, train_loader, device, mean, std, optimizer,
                              **({'loss_name': 'mse'} if config.get('loss') == 'mse' else {}))
        if distributed:
            # e3nn BatchNorm is local to each GPU. Evaluate one consistent copy
            # of its running buffers, without DDP forward collectives on uneven shards.
            for buffer in raw_model.buffers():
                dist.broadcast(buffer, src=0)
        metrics = {'epoch': step, 'train': train_metrics,
                   'validation': epoch(raw_model, val_loader, device, mean, std)}
        metrics['elapsed_seconds'] = time.perf_counter() - started
        if rank == 0:
            print(json.dumps(metrics), flush=True)
        history.append(metrics)
        if metrics['validation'][selection_metric] < best:
            best = metrics['validation'][selection_metric]
            stale_epochs = 0
            if rank == 0:
                best_state = {'model_state': raw_model.state_dict(), 'config': config,
                        'atom_init': atom_init, 'feature_count': feature_count,
                        'target_mean': mean, 'target_std': std,
                        'epoch': step, 'validation': metrics['validation'],
                        'execution': execution}
                temporary_best = args.output / 'best.pt.tmp'
                torch.save(best_state, temporary_best)
                os.replace(temporary_best, checkpoint_path)
        else:
            stale_epochs += 1
        if rank == 0:
            last_state = {
                'model_state': raw_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config,
                'atom_init': atom_init,
                'feature_count': feature_count,
                'target_mean': mean,
                'target_std': std,
                'epoch': step,
                'validation': metrics['validation'],
                'execution': execution,
                'best': best,
                'stale_epochs': stale_epochs,
                'history': history,
                'torch_rng_state': torch.get_rng_state(),
            }
            temporary_last = args.output / 'last.pt.tmp'
            torch.save(last_state, temporary_last)
            os.replace(temporary_last, last_checkpoint_path)
            temporary_history = args.output / 'history.json.tmp'
            temporary_history.write_text(json.dumps(history, indent=2))
            os.replace(temporary_history, args.output / 'history.json')
        if args.patience and stale_epochs >= args.patience:
            if rank == 0:
                print(f'Early stopping after {step} epochs: validation {selection_metric} did not improve '
                  f'for {args.patience} consecutive epochs.', flush=True)
            break
    if distributed:
        dist.barrier()
    if testing:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        raw_model.load_state_dict(checkpoint['model_state'])
        metrics = epoch(raw_model, loader(testing), device, mean, std)
        if rank == 0:
            (args.output / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
            print('Test:', json.dumps(metrics))


def predict(args):
    if args.num_workers < 0:
        raise ValueError('num_workers must be nonnegative.')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint['config']
    records = load_records(args.data, labeled=False)
    dataset = AdsorptionDataset(records, checkpoint['atom_init'], config, cache_dir=args.cache_dir)
    model = build_model(config, checkpoint['feature_count']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    values = []
    with torch.inference_mode():
        for graph in graph_loader(dataset, config['batch_size'], num_workers=args.num_workers):
            prediction = model(graph.to(device))[0] * checkpoint['target_std'] + checkpoint['target_mean']
            values.extend(prediction.flatten().cpu().tolist())
    output = [{'id': r['id'], 'adsorption_energy_eV': v} for r, v in zip(records, values)]
    with args.output.open('x') as handle:
        json.dump(output, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--train', type=Path, required=True)
    training.add_argument('--validation', type=Path, required=True)
    training.add_argument('--test', type=Path)
    training.add_argument('--config', type=Path, default=ROOT / 'configs/adsorption.json')
    training.add_argument('--output', type=Path, required=True)
    pooling_options = training.add_mutually_exclusive_group()
    pooling_options.add_argument('--adsorbate-pooling', action=argparse.BooleanOptionalAction, default=None,
                                 help='Turn adsorbate mean readout on, or off with --no-adsorbate-pooling')
    pooling_options.add_argument('--pooling', choices=['mean', 'add', 'max', 'adsorbate_mean', 'local_mean'],
                          help='Override graph readout; adsorbate_mean averages only adsorbate atoms')
    training.add_argument('--z-symmetry-breaking', action='store_true',
                          help='Add a fixed Cartesian +z vector input (default off unless enabled in config)')
    force_options = training.add_mutually_exclusive_group()
    force_options.add_argument('--force-mode', choices=FORCE_MODES,
                               help='none, adsorbate only, adsorbate plus nearby substrate, or all atoms')
    force_options.add_argument('--use-mlip-forces', action='store_true',
                               help='Legacy alias for --force-mode all')
    training.add_argument('--force-radius', type=float,
                          help='Periodic distance from any adsorbate atom for adsorbate_nearby (default 3 Angstrom)')
    training.add_argument('--force-scale', type=float,
                          help='Fixed force divisor in eV/Angstrom (default 1; never fit on validation/test)')
    training.add_argument('--batch-size', type=int, help='Graphs per GPU (global batch scales with GPU count)')
    training.add_argument('--epochs', type=int, help='Override the configuration epoch limit')
    training.add_argument('--patience', type=int, default=0,
                          help='Stop after this many epochs without improvement in validation loss metric (0 disables)')
    training.add_argument('--resume', action='store_true',
                          help='Resume single-process CPU training from OUTPUT/last.pt')
    inference = sub.add_parser('predict')
    inference.add_argument('--checkpoint', type=Path, required=True)
    inference.add_argument('--data', type=Path, required=True)
    inference.add_argument('--output', type=Path, required=True)
    for command in (training, inference):
        command.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        command.add_argument('--num-workers', type=int, default=0,
                             help='Parallel graph-loading processes per GPU')
        command.add_argument('--cache-dir', type=Path,
                             help='Optional graph cache shared across workers, epochs, and runs')
    args = parser.parse_args()
    if args.command == 'predict' and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('Prediction uses one process; launch with python, not torchrun.')
    try:
        (train if args.command == 'train' else predict)(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
