from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
import numpy as np
import torch
DATA_PATH = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))

class TemporalDataset:

    def __init__(self, name, device):
        self.root = DATA_PATH / name
        self.device = device
        self.data = {}
        for split in ('train', 'test', 'valid'):
            with (self.root / (split + '.pickle')).open('rb') as handle:
                self.data[split] = pickle.load(handle)
        rows = np.vstack([self.data['train'], self.data['valid'], self.data['test']])
        if rows.ndim != 2 or rows.shape[1] != 4:
            raise ValueError('Expected point-time quadruples with shape [N, 4]')
        maxima = rows.max(axis=0)
        shape_file = self.root / 'shape.json'
        shape = json.loads(shape_file.read_text()) if shape_file.exists() else {}
        self.n_entities = int(shape.get('n_entities', max(maxima[0], maxima[2]) + 1))
        self.n_predicates = 2 * int(shape.get('n_relations', maxima[1] + 1))
        self.n_timestamps = int(shape.get('n_timestamps', maxima[3] + 1))
        with (self.root / 'to_skip.pickle').open('rb') as handle:
            self.to_skip = pickle.load(handle)

    def get_shape(self):
        return (self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps)

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        inverse = np.copy(self.data['train'])
        temporary = np.copy(inverse[:, 0])
        inverse[:, 0] = inverse[:, 2]
        inverse[:, 2] = temporary
        inverse[:, 1] += self.n_predicates // 2
        return np.vstack((self.data['train'], inverse))

    def eval(self, model, split, batch_size=500):
        examples = torch.from_numpy(self.data[split].astype('int64')).to(self.device)
        reciprocal_ranks, hits = ({}, {})
        for direction in ('rhs', 'lhs'):
            queries = examples.clone()
            if direction == 'lhs':
                queries[:, 0] = examples[:, 2]
                queries[:, 2] = examples[:, 0]
                queries[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(queries, self.to_skip[direction], self.device, batch_size=batch_size)
            reciprocal_ranks[direction] = torch.mean(1.0 / ranks).item()
            hits[direction] = torch.FloatTensor([torch.mean((ranks <= k).float()).item() for k in (1, 3, 10)])
        return (reciprocal_ranks, hits)

    def get_multimodal_data(self, modality: str='both') -> torch.Tensor:
        name_to_id = {}
        with open(str(self.root / 'ent_id')) as f:
            for line in f:
                name, i = line.rstrip('\n').split('\t')
                name_to_id[name] = int(i)
        if modality not in {'both', 'visual', 'text'}:
            raise ValueError(f'Unsupported modality={modality}')
        name_to_mmid = {}
        mmid_path = self.root / 'ent_mmid'
        if mmid_path.exists():
            with mmid_path.open() as f:
                for line in f:
                    name, m = line.rstrip('\n').split('\t')
                    name_to_mmid[name] = int(m)
        else:
            name_to_mmid = dict(name_to_id)
        vis = np.load(str(self.root / 'matrix_visual_vgg19.npy'))
        ling = np.load(str(self.root / 'matrix_linguistic_bert-base-uncased.npy'))
        visual_dim = int(vis.shape[0])
        text_dim = int(ling.shape[1])
        feat_dim = {'both': visual_dim + text_dim, 'visual': visual_dim, 'text': text_dim}[modality]
        feat = np.zeros((self.n_entities, feat_dim), dtype=np.float32)
        n_matched = 0
        for name, base_id in name_to_id.items():
            if name not in name_to_mmid:
                continue
            mmid = name_to_mmid[name]
            if modality == 'visual':
                value = vis[:, mmid]
            elif modality == 'text':
                value = ling[mmid]
            else:
                value = np.concatenate([vis[:, mmid], ling[mmid]])
            feat[base_id] = value
            n_matched += 1
        print(f'[MM] modality={modality} shape={feat.shape}, matched={n_matched}/{self.n_entities}')
        return torch.from_numpy(feat).float()
