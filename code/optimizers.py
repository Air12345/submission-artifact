from __future__ import annotations
import torch
import torch.nn.functional as F

class TKBCOptimizer:

    def __init__(self, model, emb_regularizer, temporal_regularizer, optimizer, batch_size=256, tna_gate_reg=0.0, bi_consistency_reg=0.0, bi_consistency_every=1, label_smoothing=0.0, reuse_forward_true_score=False):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.tna_gate_reg = tna_gate_reg
        self.bi_consistency_reg = bi_consistency_reg
        self.bi_consistency_every = max(1, int(bi_consistency_every))
        self.label_smoothing = max(0.0, float(label_smoothing))
        self.reuse_forward_true_score = bool(reuse_forward_true_score)

    def _cross_entropy_each(self, predictions, truth):
        if self.label_smoothing <= 0.0:
            return F.cross_entropy(predictions, truth, reduction='none')
        eps = min(self.label_smoothing, 0.5)
        log_probs = F.log_softmax(predictions, dim=1)
        nll = -log_probs.gather(1, truth.unsqueeze(1)).squeeze(1)
        return (1.0 - eps) * nll + eps * -log_probs.mean(dim=1)

    def _inverse_batch(self, batch):
        inverse = batch.clone()
        half = self.model.sizes[1] // 2
        inverse[:, 0] = batch[:, 2]
        inverse[:, 2] = batch[:, 0]
        inverse[:, 1] = torch.where(batch[:, 1] < half, batch[:, 1] + half, batch[:, 1] - half)
        return inverse

    def epoch(self, examples, device, static_sample_weights=None):
        permutation = torch.randperm(examples.shape[0])
        actual = examples[permutation, :]
        cached = self.model.has_tna_query_cache(examples.shape[0])
        weights = static_sample_weights[permutation] if static_sample_weights is not None else None
        totals = {key: 0.0 for key in ('loss', 'reg', 'cont', 'bidir', 'total')}
        batches = consistency_batches = 0
        for begin in range(0, len(actual), self.batch_size):
            batch = actual[begin:begin + self.batch_size].to(device)
            cache_ids = permutation[begin:begin + self.batch_size].to(device) if cached else None
            predictions, factors, times, _ = self.model.forward(batch, tna_cache_ids=cache_ids)
            truth = batch[:, 2]
            each = self._cross_entropy_each(predictions, truth)
            if weights is not None:
                batch_weights = weights[begin:begin + self.batch_size].to(device).detach()
                fit = (batch_weights * each).sum() / batch_weights.sum()
            else:
                fit = each.mean()
            regularization = self.emb_regularizer.forward(factors)
            temporal = self.temporal_regularizer.forward(times)
            consistency = torch.zeros_like(fit)
            compute_consistency = self.bi_consistency_reg > 0 and batches % self.bi_consistency_every == 0
            if compute_consistency:
                inverse = self._inverse_batch(batch)
                inverse_ids = self.model.tna_inverse_cache_ids(cache_ids) if cache_ids is not None else None
                forward_score = predictions.gather(1, truth.unsqueeze(1)) if self.reuse_forward_true_score else self.model.score(batch, tna_cache_ids=cache_ids)
                consistency = F.smooth_l1_loss(forward_score, self.model.score(inverse, tna_cache_ids=inverse_ids))
                consistency_batches += 1
            loss = fit + regularization + temporal
            if self.tna_gate_reg > 0:
                coefficient_reg = (self.model.tna_gate ** 2).mean() + (self.model.tna_dir_gate ** 2).mean() + (self.model.tna_rel_gate.weight ** 2).mean()
                loss = loss + self.tna_gate_reg * coefficient_reg
            if compute_consistency:
                loss = loss + self.bi_consistency_reg * self.bi_consistency_every * consistency
            self.optimizer.zero_grad()
            loss = loss.to(device)
            loss.backward()
            for parameter in self.model.parameters():
                if parameter.grad is not None and torch.isnan(parameter.grad).any():
                    parameter.grad = torch.nan_to_num(parameter.grad)
            self.optimizer.step()
            for key, value in (('loss', fit), ('reg', regularization), ('cont', temporal), ('bidir', consistency), ('total', loss)):
                totals[key] += value.item()
            batches += 1
        return {key: value / max(consistency_batches if key == 'bidir' else batches, 1) for key, value in totals.items()}
