from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, default=PACKAGE_ROOT / 'data')
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--structure_from', type=Path, help='Reuse a completed structural run containing STRUCTURE_REPORT.json.')
    parser.add_argument('--evaluate_test', action='store_true', help='Evaluate test once after validation selection.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if 'mcc' not in config:
        parser.error('Use a main-method configuration with an mcc section.')
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mcc_output = output / 'mcc'
    if mcc_output.exists() and any(mcc_output.iterdir()):
        parser.error('MCC output already exists; use a fresh run_dir to preserve results.')
    structure_root = args.structure_from.resolve() if args.structure_from else output / 'structure'
    if args.structure_from is None:
        subprocess.run([sys.executable, '-B', str(PACKAGE_ROOT / 'code/structure_pipeline.py'), '--config', str(args.config.resolve()), '--data_root', str(args.data_root.resolve()), '--run_dir', str(structure_root), '--gpu', str(args.gpu), '--seed', str(args.seed)], check=True, cwd=PACKAGE_ROOT)
    report = json.loads((structure_root / 'STRUCTURE_REPORT.json').read_text())
    recorded = json.loads((structure_root / 'RUN_CONFIG.json').read_text())
    from structure_pipeline import comparable_structure_config
    structure, stage2 = (config['structure'], config['stage2'])
    if recorded['dataset'] != config['dataset'] or recorded['seed'] != args.seed or recorded['config']['rank'] != config['rank'] or (comparable_structure_config(recorded['effective_structure']) != comparable_structure_config(structure)):
        raise ValueError('Structural run does not match the requested dataset, seed and configuration.')
    if recorded['effective_stage2']['heldout_ratio'] != stage2['heldout_ratio']:
        raise ValueError('Structural run has a different A/B split ratio.')
    os.environ['TSMC_DATA_ROOT'] = report['runtime_data_root']
    import numpy as np
    import torch
    from datasets import TemporalDataset
    from candidate_cache import build_structure, cache_direction, load_array, safe_load, set_seed
    from candidate_cache import merge_packs, avg_metrics_from_ranks
    from stage2.mcc import build_train_only_filters, collect_scores, metrics_for_alpha, parse_grid, select_global_alpha, train_mm_scorer
    mcc_output.mkdir(parents=True)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(structure.get('tf32', False))
    options = SimpleNamespace(**config['mcc'], rank=config['rank'], topk=stage2['topk'], cache_batch_size=stage2['cache_batch_size'], max_queries_per_direction=0, tna_k=structure['tna_k'], tna_tau=structure['tna_tau'], rt_film_hidden=structure['rt_film_hidden'], rt_film_dropout=structure['rt_film_dropout'], rt_film_scale=structure['rt_film_scale'], rt_film_mul_scale=structure['rt_film_mul_scale'])
    dataset = TemporalDataset(config['dataset'], device)
    split = report['split']
    train_a = load_array(Path(report['runtime_data_root']) / split['train_a_name'] / 'train.pickle')
    train_b = load_array(Path(split['train_b_path']))
    training_filters = build_train_only_filters(dataset.data['train'], dataset.n_predicates)

    def cache(model, examples, label, filters=None):
        pack = merge_packs([cache_direction(options, dataset, model, examples, label, direction, device, filters_override=filters[direction] if filters is not None else None) for direction in ('rhs', 'lhs')])
        torch.save(pack, mcc_output / (label + '_top' + str(options.topk) + '.pt'))
        return pack
    teacher = build_structure(options, dataset, Path(report['train_a_checkpoint']), train_a, device, 'trainA')
    train_pack = cache(teacher, train_a, 'trainA', training_filters)
    selection_pack = cache(teacher, train_b, 'trainB', training_filters)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_json(mcc_output / 'CONFIG.json', {'config': config, 'seed': args.seed, 'mm_seed': args.seed + 104729, 'training_filters': 'original training facts only, exact timestamp', 'mm_fit': 'Train-A', 'mm_epoch_selection': 'Train-B', 'alpha_selection': 'official validation', 'test_selection': False})
    set_seed(args.seed + 104729)
    model, best, _ = train_mm_scorer(options, dataset, train_pack, selection_pack, mcc_output, device)
    full = build_structure(options, dataset, Path(report['full_checkpoint']), dataset.data['train'].astype(np.int64), device, 'full')
    valid_pack = cache(full, dataset.data['valid'], 'valid')
    valid_scores, _ = collect_scores(model, valid_pack, options.mm_eval_batch_size, device)
    selected, grid = select_global_alpha(valid_pack, valid_scores, parse_grid(options.fusion_alpha_grid))
    save_json(mcc_output / 'VALID_SELECTION.json', {'selected': selected, 'grid': grid, 'mm_epoch': best['epoch']})
    summary = {'protocol': 'dual-modality MCC, fixed validation-selected alpha', 'dataset': config['dataset'], 'seed': args.seed, 'mm_epoch': best['epoch'], 'valid': selected, 'mm_parameters': sum((p.numel() for p in model.parameters())), 'structural_parameters': sum((p.numel() for p in full.parameters())), 'test_evaluated': False}
    if args.evaluate_test:
        test_pack = cache(full, dataset.data['test'], 'test')
        test_scores, _ = collect_scores(model, test_pack, options.mm_eval_batch_size, device)
        summary['struct_test'] = avg_metrics_from_ranks(test_pack['exact_rank'])
        summary['test'] = metrics_for_alpha(test_pack, test_scores, selected['alpha'])
        summary['test_evaluated'] = True
    save_json(mcc_output / 'FINAL_SUMMARY.json', summary)
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
