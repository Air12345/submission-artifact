from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets import TemporalDataset
from candidate_cache import avg_metrics_from_ranks, row_zscore
from candidate_cache import safe_load

def build_train_only_filters(train: np.ndarray, n_predicates: int) -> Dict[str, Dict[Tuple[int, int, int], List[int]]]:
    base_relations = n_predicates // 2
    rhs = defaultdict(set)
    lhs = defaultdict(set)
    for subject, relation, obj, timestamp in train[:, :4]:
        s, r, o, t = map(int, (subject, relation, obj, timestamp))
        rhs[s, r, t].add(o)
        lhs[o, r + base_relations, t].add(s)
    return {'rhs': {key: sorted(values) for key, values in rhs.items()}, 'lhs': {key: sorted(values) for key, values in lhs.items()}}

def batch_from_pack(pack: Dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (pack['queries'][indices].to(device), pack['candidate_ids'][indices].to(device), pack['target_pos'][indices].to(device))

def ranks_from_scores(scores: torch.Tensor, pack: Dict[str, torch.Tensor]) -> torch.Tensor:
    positions = pack['target_pos']
    hit = positions >= 0
    ranks = pack['exact_rank'].clone().long()
    if bool(hit.any()):
        target = scores[hit].gather(1, positions[hit].view(-1, 1))
        ranks[hit] = (scores[hit] >= target).sum(dim=1).long()
    return ranks

def metrics_for_alpha(pack: Dict[str, torch.Tensor], mm_scores: torch.Tensor, alpha: float) -> Dict[str, float]:
    fused = row_zscore(pack['struct_scores']) + float(alpha) * row_zscore(mm_scores)
    ranks = pack['exact_rank'].clone().long() if float(alpha) == 0.0 else ranks_from_scores(fused, pack)
    metrics = avg_metrics_from_ranks(ranks)
    baseline = avg_metrics_from_ranks(pack['exact_rank'])
    metrics.update({'struct_mrr': baseline['mrr'], 'gain': metrics['mrr'] - baseline['mrr'], 'alpha': float(alpha), 'target_in_topk': float((pack['target_pos'] >= 0).float().mean())})
    return metrics

def mm_only_metrics(pack: Dict[str, torch.Tensor], mm_scores: torch.Tensor) -> Dict[str, float]:
    metrics = avg_metrics_from_ranks(ranks_from_scores(mm_scores, pack))
    metrics['target_in_topk'] = float((pack['target_pos'] >= 0).float().mean())
    return metrics

def parse_grid(raw: str) -> List[float]:
    values = sorted({float(item.strip()) for item in raw.split(',') if item.strip()})
    if not values or values[0] != 0.0:
        values.insert(0, 0.0)
    return values

class RelationConditionedDualMMScorer(nn.Module):

    def __init__(self, n_relations: int, n_timestamps: int, visual_features: torch.Tensor, text_features: torch.Tensor, mm_dim: int, hidden: int, dropout: float, temperature: float, relation_scale: float) -> None:
        super().__init__()
        self.register_buffer('visual_features', visual_features.float(), persistent=False)
        self.register_buffer('text_features', text_features.float(), persistent=False)
        self.visual_projector = nn.Sequential(nn.LayerNorm(int(visual_features.shape[1])), nn.Linear(int(visual_features.shape[1]), hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, mm_dim))
        self.text_projector = nn.Sequential(nn.LayerNorm(int(text_features.shape[1])), nn.Linear(int(text_features.shape[1]), hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, mm_dim))
        self.visual_rel_scale = nn.Embedding(n_relations, mm_dim)
        self.text_rel_scale = nn.Embedding(n_relations, mm_dim)
        self.visual_rel_shift = nn.Embedding(n_relations, mm_dim)
        self.text_rel_shift = nn.Embedding(n_relations, mm_dim)
        self.visual_time = nn.Embedding(n_timestamps, mm_dim)
        self.text_time = nn.Embedding(n_timestamps, mm_dim)
        self.modality_relation = nn.Embedding(n_relations, 1)
        self.modality_time = nn.Embedding(n_timestamps, 1)
        self.modality_bias = nn.Parameter(torch.zeros(1))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / max(temperature, 0.0001)))))
        self.relation_scale = float(relation_scale)
        nn.init.zeros_(self.visual_rel_scale.weight)
        nn.init.zeros_(self.text_rel_scale.weight)
        nn.init.zeros_(self.modality_relation.weight)
        nn.init.zeros_(self.modality_time.weight)

    def _project_unique(self, heads: torch.Tensor, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_ids = torch.cat([heads, candidates.reshape(-1)], dim=0)
        unique_ids, inverse = torch.unique(all_ids, sorted=False, return_inverse=True)
        visual = F.normalize(self.visual_projector(self.visual_features[unique_ids]), dim=-1)
        text = F.normalize(self.text_projector(self.text_features[unique_ids]), dim=-1)
        head_count = heads.shape[0]
        visual_head = visual[inverse[:head_count]]
        text_head = text[inverse[:head_count]]
        visual_candidates = visual[inverse[head_count:]].view(candidates.shape[0], candidates.shape[1], -1)
        text_candidates = text[inverse[head_count:]].view(candidates.shape[0], candidates.shape[1], -1)
        return (visual_head, text_head, visual_candidates, text_candidates)

    def forward(self, queries: torch.Tensor, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_head, text_head, visual_candidates, text_candidates = self._project_unique(queries[:, 0], candidates)
        relation_ids = queries[:, 1]
        timestamp_ids = queries[:, 3]
        visual_query = F.normalize(visual_head * (1.0 + self.relation_scale * torch.tanh(self.visual_rel_scale(relation_ids))) + self.visual_rel_shift(relation_ids) + self.visual_time(timestamp_ids), dim=-1)
        text_query = F.normalize(text_head * (1.0 + self.relation_scale * torch.tanh(self.text_rel_scale(relation_ids))) + self.text_rel_shift(relation_ids) + self.text_time(timestamp_ids), dim=-1)
        visual_score = torch.sum(visual_query.unsqueeze(1) * visual_candidates, dim=-1)
        text_score = torch.sum(text_query.unsqueeze(1) * text_candidates, dim=-1)
        modality_weight = torch.sigmoid(self.modality_bias + self.modality_relation(relation_ids).squeeze(1) + self.modality_time(timestamp_ids).squeeze(1))
        score = modality_weight.unsqueeze(1) * visual_score + (1.0 - modality_weight).unsqueeze(1) * text_score
        return (score * self.logit_scale.exp().clamp(max=100.0), modality_weight)

@torch.no_grad()
def collect_scores(model: RelationConditionedDualMMScorer, pack: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    model.eval()
    score_parts = []
    weights = []
    for begin in range(0, pack['queries'].shape[0], batch_size):
        indices = torch.arange(begin, min(begin + batch_size, pack['queries'].shape[0]))
        queries, candidates, _ = batch_from_pack(pack, indices, device)
        scores, modality_weight = model(queries, candidates)
        score_parts.append(scores.cpu().float())
        weights.append(modality_weight.cpu().float())
    weights_tensor = torch.cat(weights)
    return (torch.cat(score_parts), {'visual_weight_mean': float(weights_tensor.mean()), 'visual_weight_std': float(weights_tensor.std(unbiased=False))})

def select_global_alpha(pack: Dict[str, torch.Tensor], mm_scores: torch.Tensor, values: Iterable[float]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rows = [metrics_for_alpha(pack, mm_scores, alpha) for alpha in values]
    return (max(rows, key=lambda row: (row['mrr'], -row['alpha'])), rows)

def save_mm_checkpoint(path: Path, model: RelationConditionedDualMMScorer, epoch: int, train_b: Dict[str, float], model_config: Dict) -> None:
    torch.save({'epoch': epoch, 'model': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'train_b': train_b, 'model_config': model_config}, path)

def train_mm_scorer(args: argparse.Namespace, dataset: TemporalDataset, train_a_pack: Dict[str, torch.Tensor], train_b_pack: Dict[str, torch.Tensor], output: Path, device: torch.device) -> Tuple[RelationConditionedDualMMScorer, Dict, torch.Tensor]:
    visual = dataset.get_multimodal_data('visual').to(device)
    text = dataset.get_multimodal_data('text').to(device)
    model_config = {'mm_dim': args.mm_dim, 'hidden': args.mm_hidden, 'dropout': args.mm_dropout, 'temperature': args.temperature, 'relation_scale': args.relation_scale}
    model = RelationConditionedDualMMScorer(dataset.n_predicates, dataset.n_timestamps, visual, text, args.mm_dim, args.mm_hidden, args.mm_dropout, args.temperature, args.relation_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mm_lr, weight_decay=args.mm_weight_decay)
    train_indices = torch.where(train_a_pack['target_pos'] >= 0)[0]
    alpha_values = parse_grid(args.mm_select_alpha_grid)
    best_mrr = -1.0
    best = None
    history = []
    checkpoint_path = output / 'best_dual_mm_scorer.pt'
    for epoch in range(1, args.mm_epochs + 1):
        model.train()
        permutation = train_indices[torch.randperm(train_indices.numel())]
        total_loss = 0.0
        total_visual_weight = 0.0
        batches = 0
        started = time.time()
        for begin in range(0, permutation.numel(), args.mm_batch_size):
            indices = permutation[begin:begin + args.mm_batch_size]
            queries, candidates, targets = batch_from_pack(train_a_pack, indices, device)
            scores, modality_weight = model(queries, candidates)
            loss = F.cross_entropy(scores, targets)
            if args.pairwise_weight > 0 and scores.shape[1] > 1:
                positive = scores.gather(1, targets.view(-1, 1))
                negative_mask = torch.ones_like(scores, dtype=torch.bool)
                negative_mask.scatter_(1, targets.view(-1, 1), False)
                negatives = scores.masked_fill(~negative_mask, -1000000000.0).topk(k=min(args.pairwise_negatives, scores.shape[1] - 1), dim=1).values
                loss = loss + args.pairwise_weight * F.relu(args.pairwise_margin - positive + negatives).mean()
            if args.modality_balance > 0:
                loss = loss + args.modality_balance * (modality_weight.mean() - 0.5).square()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            total_visual_weight += float(modality_weight.mean().detach().cpu())
            batches += 1
        row = {'epoch': epoch, 'loss': total_loss / max(batches, 1), 'visual_weight': total_visual_weight / max(batches, 1), 'logit_scale': float(model.logit_scale.exp().detach().cpu()), 'seconds': time.time() - started}
        if epoch % args.mm_valid_freq == 0 or epoch == args.mm_epochs:
            train_b_scores, modal_stats = collect_scores(model, train_b_pack, args.mm_eval_batch_size, device)
            train_b, grid_rows = select_global_alpha(train_b_pack, train_b_scores, alpha_values)
            row['train_b'] = train_b
            row['train_b_mm_only'] = mm_only_metrics(train_b_pack, train_b_scores)
            row['modality'] = modal_stats
            row['alpha_grid'] = grid_rows
            print('[DUAL_MM] epoch={} loss={:.6f} trainB_mrr={:.6f} gain={:+.6f} alpha={:.4f} mm_only={:.6f} visual={:.4f}'.format(epoch, row['loss'], train_b['mrr'], train_b['gain'], train_b['alpha'], row['train_b_mm_only']['mrr'], modal_stats['visual_weight_mean']), flush=True)
            if train_b['mrr'] > best_mrr:
                best_mrr = train_b['mrr']
                best = {'epoch': epoch, 'train_b': train_b, 'model_config': model_config}
                save_mm_checkpoint(checkpoint_path, model, epoch, train_b, model_config)
                torch.save(train_b_scores, output / 'best_trainB_mm_scores.pt')
                print(f'[DUAL_MM_CHECKPOINT] epoch={epoch} trainB_mrr={best_mrr:.6f}', flush=True)
        history.append(row)
        (output / 'mm_history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    if best is None:
        raise RuntimeError('No multimodal scorer checkpoint was selected')
    checkpoint = safe_load(checkpoint_path, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_b_scores, _ = collect_scores(model, train_b_pack, args.mm_eval_batch_size, device)
    return (model, best, train_b_scores)
