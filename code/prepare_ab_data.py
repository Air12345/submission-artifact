from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
METADATA_FILES = ('to_skip.pickle', 'ent_id', 'ent_mmid', 'rel_id', 'ts_id', 'probas.pickle', 'matrix_linguistic_bert-base-uncased.npy', 'matrix_visual_vgg19.npy')

def load_pickle(path: Path) -> np.ndarray:
    with path.open('rb') as handle:
        return pickle.load(handle).astype('int64')

def save_pickle(path: Path, value: np.ndarray) -> None:
    with path.open('wb') as handle:
        pickle.dump(value, handle)

def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()

def link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)

def infer_shape(train: np.ndarray, valid: np.ndarray, test: np.ndarray) -> dict[str, int]:
    all_rows = np.vstack([train, valid, test])
    return {'n_entities': int(max(all_rows[:, 0].max(), all_rows[:, 2].max()) + 1), 'n_relations': int(all_rows[:, 1].max() + 1), 'n_timestamps': int(all_rows[:, 3].max() + 1)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data_root', type=Path, required=True)
    parser.add_argument('--runtime_data_root', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--heldout_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_data_root.resolve() / args.dataset
    runtime_root = args.runtime_data_root.resolve()
    output = args.output_dir.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    train = load_pickle(source / 'train.pickle')
    valid = load_pickle(source / 'valid.pickle')
    test = load_pickle(source / 'test.pickle')
    shape = infer_shape(train, valid, test)
    full_view = runtime_root / args.dataset
    if not full_view.exists():
        link(source, full_view)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(train.shape[0])
    n_b = max(1, min(train.shape[0] - 1, round(train.shape[0] * args.heldout_ratio)))
    b_indices = np.sort(order[:n_b])
    a_indices = np.sort(order[n_b:])
    train_a = train[a_indices]
    train_b = train[b_indices]
    train_a_name = f'{args.dataset}__paper_trainA_seed{args.seed}'
    train_a_view = runtime_root / train_a_name
    train_a_view.mkdir(parents=True, exist_ok=True)
    save_pickle(train_a_view / 'train.pickle', train_a)
    save_pickle(train_a_view / 'valid.pickle', valid)
    save_pickle(train_a_view / 'test.pickle', valid)
    (train_a_view / 'shape.json').write_text(json.dumps(shape, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename in METADATA_FILES:
        source_file = source / filename
        if source_file.exists():
            link(source_file, train_a_view / filename)
    train_b_path = output / 'train_b.pickle'
    save_pickle(train_b_path, train_b)
    np.savez_compressed(output / 'split_indices.npz', train_a_indices=a_indices, train_b_indices=b_indices)
    metadata = {'dataset': args.dataset, 'seed': args.seed, 'heldout_ratio': args.heldout_ratio, 'train_rows': int(train.shape[0]), 'train_a_rows': int(train_a.shape[0]), 'train_b_rows': int(train_b.shape[0]), 'train_a_name': train_a_name, 'train_b_path': str(train_b_path), 'train_a_sha256': digest(train_a), 'train_b_sha256': digest(train_b), 'shape': shape, 'official_test_reserved': True}
    meta_path = output / 'split_meta.json'
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
