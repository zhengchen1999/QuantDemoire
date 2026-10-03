#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["kaggle==2.2.4"]
# ///
"""Fetch UHDM test data or a seeded training subset from the public mirror.

Run with `uv run --script script/download_uhdm.py --output data/UHDM-full/test`.
Use `--split train --count 200 --seed 123 --output data/UHDM-paper-calibration/train`
to sample calibration pairs from the complete 4,500-pair training list.
The authors' download links and mirror provenance are saved in the manifest.
This does not claim cryptographic identity to the unavailable author archive.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import random
import tempfile
import threading
import time
import zipfile

DATASET = 'soumikrakshit/uhdm-dataset'
_local = threading.local()


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    if not hasattr(_local, 'api'):
        _local.api = KaggleApi()
        _local.api._print_dataset_url_and_license = lambda *args, **kwargs: None
    return _local.api


def listing(split):
    prefix = f'{split}/{split}/'
    records, token = [], None
    while True:
        response = api().dataset_list_files(DATASET, page_token=token, page_size=200)
        page = [x.to_dict() for x in response.dataset_files]
        records.extend(x for x in page if x['name'].startswith(prefix))
        token = response.next_page_token
        # The service lists paths lexicographically; all test files precede train.
        if not token or any(x['name'] > prefix and not x['name'].startswith(prefix) for x in page):
            break
    pairs = 500 if split == 'test' else 4500
    names = [r['name'] for r in records]
    ground_truth = {name.removesuffix('_gt.jpg') for name in names if name.endswith('_gt.jpg')}
    inputs = {name.removesuffix('_moire.jpg') for name in names if name.endswith('_moire.jpg')}
    valid = len(names)==2*pairs and len(set(names))==len(names) and len(ground_truth)==pairs and ground_truth==inputs
    if split == 'test':
        valid &= ground_truth == {f'{prefix}{i:04d}' for i in range(500)}
    if not valid:
        raise RuntimeError(f'Unexpected mirror split: {len(names)} files; expected exactly {pairs} pairs')
    for record in records:
        relative = Path(record['name'].removeprefix(prefix))
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe mirror file path')
        record['relative_path'] = relative.as_posix()
    return sorted(records, key=lambda x: x['name'])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(record, output, reuse):
    relative = Path(record['relative_path'])
    name = relative.name
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = record['totalBytes']
    origin = 'existing'
    if not target.exists() or target.stat().st_size != expected:
        cached = next((root / relative for root in reuse
                       if (root / relative).is_file() and (root / relative).stat().st_size == expected), None)
        if cached is not None:
            tmp = target.with_suffix('.tmp')
            shutil.copyfile(cached, tmp)
            tmp.replace(target)
            origin = str(cached.resolve())
        else:
            for attempt in range(5):
                try:
                    with tempfile.TemporaryDirectory(prefix='.download-', dir=output) as directory:
                        api().dataset_download_file(DATASET, record['name'], path=directory, quiet=True, force=True)
                        files = list(Path(directory).iterdir())
                        if len(files) != 1:
                            raise RuntimeError(f'Unexpected download contents for {name}')
                        file = files[0]
                        if zipfile.is_zipfile(file):
                            with zipfile.ZipFile(file) as archive:
                                if archive.namelist() != [name]:
                                    raise RuntimeError(f'Unexpected archive contents for {name}')
                                with archive.open(name) as src, (Path(directory) / name).open('wb') as dst:
                                    shutil.copyfileobj(src, dst)
                            file = Path(directory) / name
                        if file.stat().st_size != expected:
                            raise RuntimeError(f'Size mismatch for {name}')
                        file.replace(target)
                    origin = 'downloaded'
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 8))
    return {**record, 'local_name': relative.as_posix(), 'sha256': digest(target), 'origin': origin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('test','train'), default='test')
    parser.add_argument('--count', type=int, help='random training pair subset; test is always complete')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--reuse', type=Path, nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    if args.count is not None and (args.split != 'train' or not 1 <= args.count <= 4500):
        parser.error('--count is supported only for 1–4500 training pairs')
    if args.output is None:
        args.output = Path('data/UHDM-full')/args.split
    args.output.mkdir(parents=True, exist_ok=True)
    records = listing(args.split)
    pool = records
    if args.count is not None:
        pair_names = sorted(x['name'].removesuffix('_gt.jpg') for x in records if x['name'].endswith('_gt.jpg'))
        selected = set(random.Random(args.seed).sample(pair_names, args.count))
        records = [x for x in records if x['name'].removesuffix('_gt.jpg').removesuffix('_moire.jpg') in selected]
    pairs = len(records)//2
    wanted = {record['relative_path'] for record in records}
    extra = {file.relative_to(args.output).as_posix() for file in args.output.rglob('*.jpg')} - wanted
    if extra:
        raise RuntimeError('Output contains images outside this selection; use a separate output directory')
    total = sum(x['totalBytes'] for x in records)
    payload = dict(dataset=DATASET, mirror_url=f'https://www.kaggle.com/datasets/{DATASET}',
        official_download_script='https://github.com/CVMI-Lab/UHDM/blob/main/scripts/download_data.sh',
        official_file_id='1bAZRuDq_bxZgxvnGtLwORV9pFKq2_kM_' if args.split=='test' else '1NiTohc65FZZXvmFpI80OTczp5Vc6EGDe',
        split_source='https://arxiv.org/html/2207.09935',
        official_archive_hash_verified=False, expected_pairs=pairs, bytes=total,
        split=args.split, pool_pairs=len(pool)//2, selection_seed=args.seed if args.count else None,
        complete=False, remote_files=records)
    manifest = args.output / 'manifest.json'
    def save():
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(manifest)
    save()
    print(f'Listed {pairs} {args.split} pairs from {len(pool)//2}, {total / 1e9:.3f} GB', flush=True)
    if args.list_only:
        return
    missing = sum(x['totalBytes'] for x in records if not (args.output / x['relative_path']).exists())
    if shutil.disk_usage(args.output).free < missing + 2 * 1024**3:
        raise RuntimeError('Not enough free disk space including download staging')
    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, x, args.output, args.reuse): x['name'] for x in records}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append(dict(name=futures[future], error=str(error)))
            count = len(completed) + len(failures)
            if count % 40 == 0 or count == len(records):
                payload.update(files=sorted(completed, key=lambda x: x['name']), failures=failures)
                save()
                print(f'{count}/{len(records)} files, {len(failures)} failures', flush=True)
    payload['complete'] = len(completed) == len(records) and not failures
    save()
    if not payload['complete']:
        raise SystemExit('Download incomplete; rerun to resume')
    print(f'Complete: all {pairs} pairs have expected sizes and recorded SHA256 hashes', flush=True)


if __name__ == '__main__':
    main()
