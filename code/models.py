from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

class TKBCModel(nn.Module):

    def get_ranking(self, queries, filters, device, batch_size=1000, chunk_size=-1):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            for begin in range(0, self.sizes[2], chunk_size):
                rhs = self.get_rhs(begin, chunk_size)
                for row in range(0, len(queries), batch_size):
                    batch = queries[row:row + batch_size]
                    scores = self.get_queries(batch) @ rhs
                    targets = self.score(batch)
                    if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
                        raise FloatingPointError('Nonfinite structural scores')
                    for i, query in enumerate(batch):
                        key = (query[0].item(), query[1].item(), query[3].item())
                        excluded = list(filters.get(key, [])) + [query[2].item()]
                        selected = [int(x - begin) for x in excluded if begin <= x < begin + chunk_size]
                        if selected:
                            scores[i, torch.as_tensor(selected, device=scores.device)] = -1000000.0
                    ranks[row:row + batch_size] += (scores >= targets).float().sum(dim=1).cpu()
        return ranks

class TeRDy(TKBCModel):

    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, train_events, tna_k=10, tna_tau=1.0, rt_film_hidden=128, rt_film_dropout=0.05, rt_film_scale=0.2, rt_film_mul_scale=0.1, init_size=0.01):
        super(TeRDy, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.pi = 3.141592653589793
        self.embeddings = nn.ModuleList([nn.Embedding(sizes[0], rank, sparse=False), nn.Embedding(sizes[1], 2 * rank, sparse=False), nn.Embedding(sizes[3], 2 * rank, sparse=False)])
        for embedding in self.embeddings:
            embedding.weight.data *= init_size
        self.rt_film_scale = float(rt_film_scale)
        self.rt_film_mul_scale = float(rt_film_mul_scale)
        self.rt_film = nn.Sequential(nn.LayerNorm(3 * rank), nn.Linear(3 * rank, max(int(rt_film_hidden), 8)), nn.GELU(), nn.Dropout(float(rt_film_dropout)), nn.Linear(max(int(rt_film_hidden), 8), 3 * rank))
        self.rt_film_dir = nn.Embedding(2, 3 * rank)
        nn.init.zeros_(self.rt_film[-1].weight)
        nn.init.zeros_(self.rt_film[-1].bias)
        nn.init.zeros_(self.rt_film_dir.weight)
        self.tna_k = tna_k
        self.tna_tau = tna_tau
        self._tna_query_cache_ready = False
        self._tna_cache_rows = 0
        self.tna_gate = nn.Parameter(torch.zeros(rank))
        self.tna_dir_gate = nn.Parameter(torch.zeros(2, rank))
        self.tna_rel_gate = nn.Embedding(sizes[1], rank)
        nn.init.zeros_(self.tna_rel_gate.weight)
        self._build_tna_index(train_events)

    def _build_tna_index(self, train_events):
        import numpy as _np
        n_ent = self.sizes[0]
        n_rel = self.sizes[1]
        times = [[] for _ in range(n_ent)]
        nbrs = [[] for _ in range(n_ent)]
        rels = [[] for _ in range(n_ent)]
        ev = train_events.detach().cpu().numpy() if torch.is_tensor(train_events) else _np.asarray(train_events)
        for a, r, b, t in ev:
            a, r, b, t = (int(a), int(r), int(b), int(t))
            times[a].append(t)
            nbrs[a].append(b)
            rels[a].append(r)
            times[b].append(t)
            nbrs[b].append(a)
            rels[b].append(r + n_rel // 2)
        self.tna_times, self.tna_nbrs, self.tna_rels = ([], [], [])
        for e in range(n_ent):
            if len(times[e]) == 0:
                self.tna_times.append(_np.zeros(0, _np.int64))
                self.tna_nbrs.append(_np.zeros(0, _np.int64))
                self.tna_rels.append(_np.zeros(0, _np.int64))
                continue
            order = _np.argsort(times[e])
            self.tna_times.append(_np.asarray(times[e], _np.int64)[order])
            self.tna_nbrs.append(_np.asarray(nbrs[e], _np.int64)[order])
            self.tna_rels.append(_np.asarray(rels[e], _np.int64)[order])

    def prepare_tna_query_cache(self, examples, device=None):
        import numpy as _np
        rows = examples.detach().cpu().numpy() if torch.is_tensor(examples) else _np.asarray(examples)
        if rows.ndim != 2 or rows.shape[1] < 4:
            raise ValueError('TNA query cache expects [N,4] training rows.')
        keys, row_to_unique = _np.unique(rows[:, [0, 3]].astype(_np.int64, copy=False), axis=0, return_inverse=True)
        n_unique = int(keys.shape[0])
        k = int(self.tna_k)
        nbr = _np.zeros((n_unique, k), dtype=_np.int32)
        rel = _np.zeros((n_unique, k), dtype=_np.int32)
        decay = _np.zeros((n_unique, k), dtype=_np.float32)
        mask = _np.zeros((n_unique, k), dtype=_np.bool_)
        unique_entities, entity_starts = _np.unique(keys[:, 0], return_index=True)
        entity_ends = _np.r_[entity_starts[1:], n_unique]
        for ent, begin, end in zip(unique_entities, entity_starts, entity_ends):
            hist_times = self.tna_times[int(ent)]
            if hist_times.shape[0] == 0:
                continue
            query_times = keys[begin:end, 1]
            positions = _np.searchsorted(hist_times, query_times, side='left')
            row_ids = _np.arange(begin, end)
            for column in range(k):
                hist_index = positions - k + column
                valid = hist_index >= 0
                if not valid.any():
                    continue
                out_rows = row_ids[valid]
                selected = hist_index[valid]
                nbr[out_rows, column] = self.tna_nbrs[int(ent)][selected]
                rel[out_rows, column] = self.tna_rels[int(ent)][selected]
                dt = query_times[valid] - hist_times[selected]
                decay[out_rows, column] = _np.exp(-dt.astype(_np.float32) / self.tna_tau)
                mask[out_rows, column] = True
        target_device = device or self.tna_gate.device
        self._tna_cache_row_to_unique = torch.from_numpy(row_to_unique.astype(_np.int32, copy=False)).to(target_device)
        self._tna_cache_nbr = torch.from_numpy(nbr).to(target_device)
        self._tna_cache_rel = torch.from_numpy(rel).to(target_device)
        self._tna_cache_decay = torch.from_numpy(decay).to(target_device)
        self._tna_cache_mask = torch.from_numpy(mask).to(target_device)
        self._tna_cache_rows = int(rows.shape[0])
        self._tna_query_cache_ready = True
        self._tna_cache_inverse_rows = None
        if rows.shape[0] % 2 == 0:
            half = rows.shape[0] // 2
            forward = rows[:half]
            inverse = rows[half:]
            reciprocal_ok = _np.array_equal(inverse[:, 0], forward[:, 2]) and _np.array_equal(inverse[:, 2], forward[:, 0]) and _np.array_equal(inverse[:, 3], forward[:, 3]) and _np.array_equal(inverse[:, 1], forward[:, 1] + self.sizes[1] // 2)
            if reciprocal_ok:
                inverse_rows = _np.r_[_np.arange(half, rows.shape[0], dtype=_np.int32), _np.arange(0, half, dtype=_np.int32)]
                self._tna_cache_inverse_rows = torch.from_numpy(inverse_rows).to(target_device)
        cache_bytes = sum((tensor.numel() * tensor.element_size() for tensor in (self._tna_cache_row_to_unique, self._tna_cache_nbr, self._tna_cache_rel, self._tna_cache_decay, self._tna_cache_mask)))
        return {'rows': self._tna_cache_rows, 'unique_entity_time': n_unique, 'compression_ratio': n_unique / max(self._tna_cache_rows, 1), 'cache_mb': cache_bytes / 1024 ** 2, 'inverse_row_map': self._tna_cache_inverse_rows is not None}

    def has_tna_query_cache(self, rows=None):
        if not self._tna_query_cache_ready:
            return False
        return rows is None or int(rows) == self._tna_cache_rows

    def tna_inverse_cache_ids(self, cache_ids):
        if self._tna_cache_inverse_rows is None:
            return None
        return self._tna_cache_inverse_rows[cache_ids]

    def _temporal_ctx(self, ent_ids: torch.Tensor, time_ids: torch.Tensor, rel_ids: torch.Tensor=None, cache_ids: torch.Tensor=None) -> torch.Tensor:
        import numpy as _np
        B = ent_ids.shape[0]
        K = self.tna_k
        dev = self.tna_gate.device
        if cache_ids is not None and self._tna_query_cache_ready:
            unique_ids = self._tna_cache_row_to_unique[cache_ids].long()
            nbr = self._tna_cache_nbr[unique_ids].long()
            rel = self._tna_cache_rel[unique_ids].long()
            decay = self._tna_cache_decay[unique_ids]
            mask = self._tna_cache_mask[unique_ids].to(decay.dtype)
        else:
            ent_np = ent_ids.detach().cpu().numpy()
            time_np = time_ids.detach().cpu().numpy()
            nbr = _np.zeros((B, K), _np.int64)
            rel = _np.zeros((B, K), _np.int64)
            decay = _np.zeros((B, K), _np.float32)
            mask = _np.zeros((B, K), _np.float32)
            for i in range(B):
                t_arr = self.tna_times[ent_np[i]]
                if t_arr.shape[0] == 0:
                    continue
                pos = int(_np.searchsorted(t_arr, int(time_np[i]), side='left'))
                if pos == 0:
                    continue
                start = max(0, pos - K)
                sel = slice(start, pos)
                n = pos - start
                nbr[i, K - n:] = self.tna_nbrs[ent_np[i]][sel]
                rel[i, K - n:] = self.tna_rels[ent_np[i]][sel]
                dt = int(time_np[i]) - self.tna_times[ent_np[i]][sel]
                decay[i, K - n:] = _np.exp(-dt.astype(_np.float32) / self.tna_tau)
                mask[i, K - n:] = 1.0
            nbr = torch.from_numpy(nbr).to(dev)
            rel = torch.from_numpy(rel).to(dev)
            decay = torch.from_numpy(decay).to(dev)
            mask = torch.from_numpy(mask).to(dev)
        nbr_emb = self.embeddings[0](nbr)[:, :, :self.rank]
        rel_emb = self.embeddings[1](rel)[:, :, :self.rank]
        nbr_vec = nbr_emb * rel_emb
        w = decay * mask
        denom = mask.sum(1, keepdim=True)
        ctx = (w.unsqueeze(-1) * nbr_vec).sum(1) / (denom + 1e-08)
        return ctx

    def _apply_tna(self, lhs0: torch.Tensor, ent_ids: torch.Tensor, time_ids: torch.Tensor, rel_ids: torch.Tensor, cache_ids: torch.Tensor=None) -> torch.Tensor:
        ctx = self._temporal_ctx(ent_ids, time_ids, rel_ids, cache_ids=cache_ids)
        gate = self.tna_gate
        direction = (rel_ids >= self.sizes[1] // 2).long()
        gate = gate + self.tna_dir_gate[direction]
        gate = gate + self.tna_rel_gate(rel_ids)
        return lhs0 + gate * ctx

    def _apply_relation_time_film(self, rel0: torch.Tensor, rel1: torch.Tensor, time0: torch.Tensor, rel_ids: torch.Tensor):
        direction = (rel_ids >= self.sizes[1] // 2).long()
        rel_norm = F.normalize(rel0, dim=-1)
        time_norm = F.normalize(time0, dim=-1)
        film_in = torch.cat([rel_norm, time_norm, rel_norm * time_norm], dim=-1)
        raw = self.rt_film(film_in) + self.rt_film_dir(direction)
        gamma, beta, eta = raw.chunk(3, dim=-1)
        scale = max(float(self.rt_film_scale), 0.0)
        gamma = torch.tanh(gamma) * scale
        beta = torch.tanh(beta) * scale
        rel_scale = rel0.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-06)
        rel0 = rel0 * (1.0 + gamma) + beta * rel_scale
        if eta is not None and self.rt_film_mul_scale > 0:
            mul_scale = max(float(self.rt_film_mul_scale), 0.0)
            rel1 = rel1 * (1.0 + mul_scale * torch.tanh(eta))
        return (rel0, rel1)

    @staticmethod
    def _safe_pair_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(a ** 2 + b ** 2 + 1e-12)

    def _split_entity_embedding(self, emb: torch.Tensor):
        first = emb[:, :self.rank]
        second = torch.zeros_like(first)
        return (first, second)

    def _split_entity_table(self, table: torch.Tensor):
        first = table[:, :self.rank]
        second = torch.zeros_like(first)
        return (first, second)

    def score(self, x, tna_cache_ids=None):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        lhs = self._split_entity_embedding(lhs)
        rhs = self._split_entity_embedding(rhs)
        rel = (rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi))
        time = (time[:, :self.rank], time[:, self.rank:])
        rel = self._apply_relation_time_film(rel[0], rel[1], time[0], x[:, 1])
        rt = (self._direct_relation_time_rt(rel[0], time), rel[1])
        lhs0, rhs0 = (lhs[0], rhs[0])
        lhs0 = self._apply_tna(lhs0, x[:, 0], x[:, 3], x[:, 1], cache_ids=tna_cache_ids)
        q = (lhs0 + rt[1]) * rt[0]
        base_score = torch.sum(q * rhs0, 1, keepdim=True)
        return base_score

    def forward(self, x, tna_cache_ids=None):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        lhs = self._split_entity_embedding(lhs)
        rhs = self._split_entity_embedding(rhs)
        rel = (rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi))
        time = (time[:, :self.rank], time[:, self.rank:])
        rel = self._apply_relation_time_film(rel[0], rel[1], time[0], x[:, 1])
        rt = (self._direct_relation_time_rt(rel[0], time), rel[1])
        loss_freq = rel[0].new_zeros(())
        right = self._split_entity_table(self.embeddings[0].weight)
        lhs0, rhs0, right0 = (lhs[0], rhs[0], right[0])
        lhs0 = self._apply_tna(lhs0, x[:, 0], x[:, 3], x[:, 1], cache_ids=tna_cache_ids)
        score_lhs0 = lhs0
        score_right0 = right0
        q = (score_lhs0 + rt[1]) * rt[0]
        predictions = q @ score_right0.t()
        factors = (torch.abs(lhs0), self._safe_pair_norm(rt[0], rt[1]), torch.abs(rhs0))
        out = (predictions, factors, self.embeddings[2].weight, loss_freq)
        return out

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        rhs = self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:, :self.rank].transpose(0, 1)
        return rhs

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = self._split_entity_embedding(lhs)
        rel = (rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi))
        time = (time[:, :self.rank], time[:, self.rank:])
        rel = self._apply_relation_time_film(rel[0], rel[1], time[0], queries[:, 1])
        rt = (self._direct_relation_time_rt(rel[0], time), rel[1])
        lhs0 = lhs[0]
        lhs0 = self._apply_tna(lhs0, queries[:, 0], queries[:, 3], queries[:, 1])
        q = (lhs0 + rt[1]) * rt[0]
        return q

    def _direct_relation_time_rt(self, rel0, time):
        return rel0 * time[1]
