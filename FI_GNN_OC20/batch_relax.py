"""Parallel, resumable MLIP relaxation of adsorption structure records."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Dataset.adsorption_records import load_records
from relax_structures import relax_structure
from mlip_calculators import create_calculator


_WORKER_CALCULATOR = None
_WORKER_RELAX_OPTIONS = None


def _json_default(value):
    """Convert NumPy-style scalar/array values to standard JSON values."""
    tolist = getattr(value, 'tolist', None)
    if callable(tolist):
        return tolist()
    item = getattr(value, 'item', None)
    if callable(item):
        return item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _initialize_worker(mlip, device, cache_dir, threads_per_worker, relax_options):
    """Load one MLIP per worker and keep it alive for all assigned structures."""
    global _WORKER_CALCULATOR, _WORKER_RELAX_OPTIONS
    import torch

    torch.set_num_threads(threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    calculator_options = {}
    if cache_dir is not None and mlip == 'uma':
        calculator_options['cache_dir'] = cache_dir
    _WORKER_CALCULATOR = create_calculator(mlip, device=device, **calculator_options)
    _WORKER_RELAX_OPTIONS = relax_options


def _relax_in_worker(index, record):
    if _WORKER_CALCULATOR is None:
        raise RuntimeError('Relaxation worker was not initialized.')
    relaxed = relax_structure(record, _WORKER_CALCULATOR, **_WORKER_RELAX_OPTIONS)
    return index, relaxed


def _atomic_json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, separators=(',', ':'), default=_json_default)
    temporary.replace(path)


def _load_jsonl_checkpoint(path, valid_ids):
    completed = {}
    if not path.exists():
        return completed
    valid_end = 0
    incomplete_tail = False
    with path.open() as handle:
        line_number = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                valid_end = handle.tell()
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f'Removing incomplete checkpoint tail starting at line '
                      f'{line_number} in {path}.', flush=True)
                incomplete_tail = True
                break
            record_id = str(record['id'])
            if record_id not in valid_ids:
                raise ValueError(f'Checkpoint contains ID {record_id!r} absent from the input data.')
            completed[record_id] = record
            valid_end = handle.tell()
    if incomplete_tail:
        # A process can be interrupted during its final write. Remove the bad
        # tail before appending, otherwise every later record would remain
        # attached to invalid JSON and could never be recovered.
        with path.open('r+') as handle:
            handle.truncate(valid_end)
    return completed


def _append_checkpoint(path, record):
    # Serialize before opening the checkpoint. A serialization error must not
    # leave half a record at the end of an otherwise valid recovery file.
    serialized = json.dumps(record, separators=(',', ':'), default=_json_default)
    with path.open('a') as handle:
        handle.write(serialized + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def _validate_records(records, expected_count=None):
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f'Expected {expected_count} structures, found {len(records)}.')
    ids = [str(record['id']) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Input contains duplicate structure IDs; safe checkpoint recovery is impossible.')
    return ids


def relax_dataset_with_checkpointing(input_path, output_path, mlip='orb', device='cpu',
                                     optimizer_cls='BFGS', fmax=0.05, max_steps=50,
                                     slab_vacuum=15.0, workers=1,
                                     threads_per_worker=1, expected_count=None,
                                     limit=None, cache_dir=None):
    """Relax a dataset with bounded process parallelism and append-only recovery."""
    if workers < 1 or threads_per_worker < 1:
        raise ValueError('workers and threads_per_worker must be positive integers.')
    if workers > 1 and not str(device).startswith('cpu'):
        raise ValueError('Multiple relaxation workers are supported only for CPU inference.')

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix('.checkpoint.jsonl')
    legacy_checkpoint = output_path.with_suffix('.checkpoint.json')

    raw_records = load_records(input_path)
    if limit is not None:
        raw_records = raw_records[:limit]
    input_ids = _validate_records(raw_records, expected_count=expected_count)
    valid_ids = set(input_ids)
    total = len(raw_records)

    # A completed output is itself the strongest restart marker. Validate it
    # before loading an expensive MLIP model.
    if output_path.exists():
        existing = load_records(output_path)
        existing_ids = _validate_records(existing, expected_count=total)
        if existing_ids != input_ids:
            raise ValueError(f'Existing output does not match input ordering/IDs: {output_path}')
        print(f'Validated completed relaxation output ({total} structures): {output_path}', flush=True)
        return existing

    completed = _load_jsonl_checkpoint(checkpoint_path, valid_ids)
    # Read checkpoints produced by the original serial implementation once,
    # then continue in the append-only JSONL format.
    if legacy_checkpoint.exists() and not completed:
        try:
            legacy_records = json.loads(legacy_checkpoint.read_text())
            for record in legacy_records:
                record_id = str(record['id'])
                if record_id not in valid_ids:
                    raise ValueError(f'Legacy checkpoint ID {record_id!r} is absent from input data.')
                completed[record_id] = record
            for record_id in input_ids:
                if record_id in completed:
                    _append_checkpoint(checkpoint_path, completed[record_id])
            print(f'Converted legacy checkpoint with {len(completed)}/{total} structures.', flush=True)
        except Exception as error:
            raise ValueError(f'Cannot safely read legacy checkpoint {legacy_checkpoint}: {error}') from error

    pending = [(index, record) for index, record in enumerate(raw_records)
               if str(record['id']) not in completed]
    if completed:
        print(f'Resuming with {len(completed)}/{total} structures already relaxed.', flush=True)
    if not pending:
        final_records = [completed[record_id] for record_id in input_ids]
        _atomic_json_write(output_path, final_records)
        checkpoint_path.unlink(missing_ok=True)
        legacy_checkpoint.unlink(missing_ok=True)
        print(f'Finalized {total} completed checkpoint records without loading the MLIP.', flush=True)
        print(f'Saved: {output_path}', flush=True)
        return final_records
    print(f'Loading {mlip} calculator in {workers} worker(s) on {device}; '
          f'{threads_per_worker} thread(s) per worker.', flush=True)

    relax_options = dict(
        optimizer_cls=optimizer_cls,
        fmax=fmax,
        max_steps=max_steps,
        slab_vacuum=slab_vacuum,
    )
    started = time.time()
    newly_completed = 0

    def record_result(index, relaxed_record):
        nonlocal newly_completed
        expected_id = input_ids[index]
        record_id = str(relaxed_record['id'])
        if record_id != expected_id:
            raise ValueError(f'Worker returned ID {record_id!r}; expected {expected_id!r}.')
        completed[record_id] = relaxed_record
        _append_checkpoint(checkpoint_path, relaxed_record)
        newly_completed += 1
        done = len(completed)
        if done % 10 == 0 or done == total:
            metadata = relaxed_record['relaxation_metadata']
            seconds_per_new = (time.time() - started) / max(newly_completed, 1)
            eta_hours = (len(pending) - newly_completed) * seconds_per_new / 3600.0
            print(f'[{done}/{total}] {record_id}: '
                  f"fmax {metadata['initial_fmax_eV_A']:.2f} -> "
                  f"{metadata['final_fmax_eV_A']:.2f} eV/A "
                  f"({metadata['steps_taken']} steps, converged={metadata['converged']}, "
                  f'ETA {eta_hours:.1f} h)', flush=True)

    if workers == 1:
        _initialize_worker(mlip, device, cache_dir, threads_per_worker, relax_options)
        for index, record in pending:
            record_result(*_relax_in_worker(index, record))
    else:
        # Keep only a small number of large structure records in IPC queues.
        # Submitting all 10k records at once can consume many GB of memory.
        iterator = iter(pending)
        inflight = {}
        max_inflight = workers * 2
        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(mlip, device, cache_dir, threads_per_worker, relax_options)) as executor:
            for _ in range(min(max_inflight, len(pending))):
                index, record = next(iterator)
                inflight[executor.submit(_relax_in_worker, index, record)] = index
            while inflight:
                finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in finished:
                    inflight.pop(future)
                    record_result(*future.result())
                    try:
                        index, record = next(iterator)
                    except StopIteration:
                        continue
                    inflight[executor.submit(_relax_in_worker, index, record)] = index

    final_records = [completed[record_id] for record_id in input_ids]
    _atomic_json_write(output_path, final_records)
    checkpoint_path.unlink(missing_ok=True)
    legacy_checkpoint.unlink(missing_ok=True)

    elapsed = time.time() - started
    converged = sum(bool(record['relaxation_metadata']['converged']) for record in final_records)
    print(f'Finished {total} structures in {elapsed / 3600.0:.2f} h; '
          f'{converged}/{total} converged.', flush=True)
    print(f'Saved: {output_path}', flush=True)
    return final_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mlip', choices=['orb', 'uma'], default='orb')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--optimizer', choices=['BFGS', 'LBFGS'], default='BFGS')
    parser.add_argument('--fmax', type=float, default=0.05)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--slab-vacuum', type=float, default=15.0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()

    relax_dataset_with_checkpointing(
        input_path=args.input,
        output_path=args.output,
        mlip=args.mlip,
        device=args.device,
        optimizer_cls=args.optimizer,
        fmax=args.fmax,
        max_steps=args.max_steps,
        slab_vacuum=args.slab_vacuum,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        expected_count=args.expected_count,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )


if __name__ == '__main__':
    main()
