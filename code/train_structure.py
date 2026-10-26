from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from datasets import TemporalDataset
from models import TSMCStructuralEncoder
from optimizers import StructuralOptimizer
from regularizers import N3, Lambda3
from candidate_cache import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset_view', required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    structure = config['structure']
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved_config = {'config': config, 'dataset_view': args.dataset_view, 'seed': args.seed}
    config_path = output / 'CONFIG.json'
    if config_path.exists() and json.loads(config_path.read_text()) != saved_config:
        raise ValueError('Existing structural run has a different configuration')
    config_path.write_text(json.dumps(saved_config, indent=2))
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(structure['tf32'])
        torch.backends.cudnn.allow_tf32 = bool(structure['tf32'])
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high' if structure['tf32'] else 'highest')
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    dataset = TemporalDataset(args.dataset_view, device)
    model = TSMCStructuralEncoder(dataset.get_shape(), config['rank'], train_events=torch.from_numpy(dataset.data['train'].astype('int64')), tna_k=structure['tna_k'], tna_tau=structure['tna_tau'], rt_film_hidden=structure['rt_film_hidden'], rt_film_dropout=structure['rt_film_dropout'], rt_film_scale=structure['rt_film_scale'], rt_film_mul_scale=structure['rt_film_mul_scale']).to(device)
    film_parameters = list(model.rt_film.parameters()) + list(model.rt_film_dir.parameters())
    neighbor_parameters = [model.tna_gate, model.tna_dir_gate] + list(model.tna_rel_gate.parameters())
    separate = {id(p) for p in film_parameters + neighbor_parameters}
    optimizer = torch.optim.Adagrad([{'params': [p for p in model.parameters() if id(p) not in separate], 'lr': structure['learning_rate']}, {'params': film_parameters, 'lr': structure['rt_film_lr']}, {'params': neighbor_parameters, 'lr': structure['tna_lr']}], lr=structure['learning_rate'])
    training = torch.from_numpy(dataset.get_train().astype('int64'))
    if structure['tna_query_cache']:
        model.prepare_tna_query_cache(training, device=device)
    direction_weights = None
    if structure['inverse_weight'] != 1.0:
        direction_weights = torch.ones(training.shape[0], dtype=torch.float)
        direction_weights[len(dataset.data['train']):] = float(structure['inverse_weight'])
    engine = StructuralOptimizer(model, N3(structure['emb_reg']), Lambda3(structure['time_reg']), optimizer, batch_size=structure['batch_size'], tna_gate_reg=structure['tna_gate_reg'], bi_consistency_reg=structure['bi_consistency_reg'], bi_consistency_every=structure['bi_consistency_every'], label_smoothing=structure['label_smoothing'], reuse_forward_true_score=structure['reuse_forward_true_score'])
    state_path = output / 'last_training_state.pt'
    start, patience, best = (0, 0, 0.0)
    history = []
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start, patience, best = (state['epoch'] + 1, state['patience'], state['best_select_score'])
        history = state['history']
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if torch.cuda.is_available() and state['cuda_rng_state_all'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng_state_all']])
    for epoch in range(start, structure['max_epochs']):
        model.train()
        losses = engine.epoch(training, device, static_sample_weights=direction_weights)
        row = {'epoch': epoch, 'losses': losses}
        if (epoch + 1) % structure['valid_freq'] == 0:
            model.eval()
            with torch.no_grad():
                mrr, hits = dataset.eval(model, 'valid', batch_size=structure['eval_batch_size'])
            average = (mrr['lhs'] + mrr['rhs']) / 2
            score = average + structure['select_lhs_weight'] * mrr['lhs']
            row['valid'] = {'mrr': average, 'direction_mrr': mrr, 'hits': {k: v.tolist() for k, v in hits.items()}, 'selection_score': score}
            if score < best:
                patience += 1
            else:
                best, patience = (score, 0)
                torch.save(model.state_dict(), output / 'best_structure.pt')
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        if (epoch + 1) % structure['state_save_freq'] == 0 or patience >= structure['patience']:
            temporary = output / 'state.tmp'
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'patience': patience, 'best_select_score': best, 'history': history, 'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(), 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temporary)
            os.replace(temporary, state_path)
        if patience >= structure['patience']:
            break
    if not (output / 'best_structure.pt').exists():
        raise RuntimeError('No validation-selected structural checkpoint was produced')
if __name__ == '__main__':
    main()
