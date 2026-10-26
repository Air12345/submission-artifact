from __future__ import annotations
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from models import TSMCStructuralEncoder
from datasets import TemporalDataset

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        return torch.load(path, map_location=device)

def make_queries(examples: np.ndarray, dataset: TemporalDataset, direction: str) -> torch.Tensor:
    queries = torch.from_numpy(examples.astype('int64'))
    if direction == 'rhs':
        return queries
    lhs = queries.clone()
    lhs[:, 0] = queries[:, 2]
    lhs[:, 2] = queries[:, 0]
    lhs[:, 1] = queries[:, 1] + dataset.n_predicates // 2
    return lhs

@torch.no_grad()
def cache_direction(args, dataset: TemporalDataset, model: TSMCStructuralEncoder, examples: np.ndarray, split: str, direction: str, device: torch.device, filters_override=None) -> Dict[str, torch.Tensor]:
    queries = make_queries(examples, dataset, direction)
    max_queries = int(getattr(args, 'max_queries_per_direction', 0))
    if max_queries > 0:
        queries = queries[:max_queries]
    filters = filters_override if filters_override is not None else dataset.to_skip[direction]
    fields: Dict[str, List[torch.Tensor]] = {key: [] for key in ('queries', 'direction', 'struct_scores', 'candidate_ids', 'target_pos', 'exact_rank')}
    direction_id = 1 if direction == 'lhs' else 0
    print(f'[CACHE_START] split={split} direction={direction} queries={len(queries)} topK={args.topk}', flush=True)
    for begin in range(0, len(queries), args.cache_batch_size):
        batch_cpu = queries[begin:begin + args.cache_batch_size]
        batch = batch_cpu.to(device)
        scores = model.forward(batch)[0]
        top_scores, top_ids, exact_rank = apply_filtered_topk(scores, batch, filters, args.topk)
        target_match = top_ids == batch[:, 2].view(-1, 1)
        target_hit = target_match.any(dim=1)
        target_pos = torch.full((batch.shape[0],), -1, dtype=torch.long, device=device)
        if bool(target_hit.any()):
            target_pos[target_hit] = target_match[target_hit].float().argmax(dim=1).long()
        fields['queries'].append(batch_cpu)
        fields['direction'].append(torch.full((batch.shape[0],), direction_id, dtype=torch.long))
        fields['struct_scores'].append(top_scores.cpu().float())
        fields['candidate_ids'].append(top_ids.cpu().long())
        fields['target_pos'].append(target_pos.cpu())
        fields['exact_rank'].append(exact_rank.cpu())
        batch_id = begin // args.cache_batch_size
        if batch_id == 0 or batch_id % 50 == 0:
            print(f'[CACHE_PROGRESS] {split}/{direction} {min(begin + args.cache_batch_size, len(queries))}/{len(queries)}', flush=True)
    return {key: torch.cat(parts, dim=0) for key, parts in fields.items()}

def row_zscore(x: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)

def apply_filtered_topk(scores: torch.Tensor, queries: torch.Tensor, filters: Dict[Tuple[int, int, int], List[int]], topk: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    filtered = scores.clone()
    for i, q in enumerate(queries):
        key = (int(q[0].item()), int(q[1].item()), int(q[3].item()))
        target = int(q[2].item())
        filter_out = [int(x) for x in filters.get(key, []) if int(x) != target]
        if filter_out:
            filtered[i, torch.as_tensor(filter_out, device=filtered.device, dtype=torch.long)] = -1000000.0
    target_scores = filtered.gather(1, queries[:, 2].view(-1, 1))
    exact_rank = (filtered >= target_scores).sum(dim=1).long()
    k = min(int(topk), filtered.shape[1])
    top_scores, top_ids = torch.topk(filtered, k=k, dim=1)
    return (top_scores, top_ids, exact_rank)

def avg_metrics_from_ranks(ranks: torch.Tensor) -> Dict[str, float]:
    ranks = ranks.float()
    return {'mrr': float((1.0 / ranks).mean().item()), 'h1': float((ranks <= 1).float().mean().item()), 'h3': float((ranks <= 3).float().mean().item()), 'h10': float((ranks <= 10).float().mean().item())}

def merge_packs(packs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([p[k] for p in packs], dim=0) for k in packs[0].keys() if k != 'seconds'}

def build_structure(args, dataset, checkpoint, train_events, device, label):
    model = TSMCStructuralEncoder(dataset.get_shape(), rank=args.rank, train_events=torch.from_numpy(train_events), tna_k=args.tna_k, tna_tau=args.tna_tau, rt_film_hidden=args.rt_film_hidden, rt_film_dropout=args.rt_film_dropout, rt_film_scale=args.rt_film_scale, rt_film_mul_scale=args.rt_film_mul_scale).to(device)
    state = safe_load(checkpoint, device)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
