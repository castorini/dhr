import random
from dataclasses import dataclass
from typing import List, Tuple

from tqdm import tqdm
import glob
import os
import json

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
import torch

from .arguments import DataArguments
from .trainer import DenseTrainer

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: DenseTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        return encoded_query, encoded_passages

class TrainTASBDataset(Dataset):
    # This is now only for msmarco-passage; since the id starts from 0. While using other datasets, this should be revised.
    def __init__(
            self,
            data_args: DataArguments,
            kd,
            dataset: datasets.Dataset,
            corpus: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: DenseTrainer = None,
    ):
        self.train_data, self.qidx_cluster = dataset
        self.corpus = corpus
        self.tok = tokenizer
        self.trainer = trainer
        self.data_args = data_args
        self.tasb_sampling = data_args.tasb_sampling
        self.kd = kd

        if self.data_args.corpus_dir is None:
            raise ValueError('You should input --corpus_dir with files split*.json')

        # if (self.data_args.train_n_passages!=2) and (self.tasb_sampling):
        #     raise ValueError('--train_n_passages should be 2 if you use tasb sampling')

        if (self.qidx_cluster is None) and (self.tasb_sampling):
            raise ValueError('You should input  --query_cluster_dir for tasb sampling')

        self.data_args = data_args
        self.total_len = len(self.train_data)
        if self.qidx_cluster:
            self.cluster_num = len(self.qidx_cluster)
        

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def output_qp(self, group, _hashed_seed):
        epoch = int(self.trainer.state.epoch)
        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['positive_pids']
        group_negatives = group['negative_pids']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg_id = group_positives[0]
        else:
            pos_psg_id = group_positives[(_hashed_seed + epoch) % len(group_positives)]
            pos_psg = self.corpus[int(pos_psg_id)]['text']
        encoded_passages.append(self.create_one_example(pos_psg))
        
        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg_pid in negs:
            neg_psg = self.corpus[int(neg_psg_pid)]['text']
            encoded_passages.append(self.create_one_example(neg_psg))
        
        return encoded_query, encoded_passages, None
    
    def output_qp_with_score(self, group, _hashed_seed):
        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        scores = []
        qids_bin_pairs = group['bin_pairs']
        bins_pairs = random.choices(qids_bin_pairs, k=1)[0]

        pairs = []
        negative_size = self.data_args.train_n_passages - 1

        for i in range(negative_size):
            bin_pairs = random.choices(bins_pairs, k=1)[0]
            pairs.append(random.choices(bin_pairs, k=1)[0])

        pos_psg_idx = int(pairs[0][0])
        pos_psg_id = group['positive_pids'][pos_psg_idx]
        pos_psg = self.corpus[int(pos_psg_id)]['text']
        encoded_passages.append(self.create_one_example(pos_psg))
        
        for pair in pairs:
            neg_psg_idx = int(pair[1])
            neg_psg_id = group['negative_pids'][neg_psg_idx]
            neg_psg = self.corpus[int(neg_psg_id)]['text']
            encoded_passages.append(self.create_one_example(neg_psg))
            scores.append(-pair[2])

        return encoded_query, encoded_passages, scores

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        _hashed_seed = hash(item + self.trainer.args.seed)
        if self.tasb_sampling:
            # make sure the same query cluster gathered in the same batch
            random.seed(self.trainer.state.global_step)
            cluster_num = random.randint(0, self.cluster_num-1)
            
            #sampling different queries in a batch
            random.seed(_hashed_seed) 
            item = random.choices(self.qidx_cluster[cluster_num]['qidx'])[0]

            group = self.train_data[item]
        else:
            group = self.train_data[item]
        
        if self.kd:            
            return self.output_qp_with_score(group, _hashed_seed)
        else:
            return self.output_qp(group, _hashed_seed)
        



class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        if len(text)==0:
            text = [0]
        encoded_text = self.tok.encode_plus(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text

class EvalDataset(Dataset):
    input_keys = ['qry_text_id', 'qry_text', 'psg_text_id', 'psg_text', 'rel']

    def __init__(self, 
                 data_args: DataArguments, 
                 dataset: datasets.Dataset, 
                 tokenizer: PreTrainedTokenizer):
        self.encode_data = dataset
        self.tok = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        qry_text_id, qry_text, psg_text_id, psg_text, rel = (self.encode_data[item][f] for f in self.input_keys)
        encoded_qry_text = self.tok.encode_plus(
            qry_text,
            max_length=self.data_args.q_max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        if len(psg_text)==0:
            psg_text = [0]
        encoded_psg_text = self.tok.encode_plus(
            psg_text,
            max_length=self.data_args.p_max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return qry_text_id, encoded_qry_text, psg_text_id, encoded_psg_text, rel


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        
        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        if features[0][2] is not None:
            scores = [[0]+f[2] for f in features]
            scores_collated = torch.tensor(scores)
        else:
            scores_collated = None

        return q_collated, d_collated, scores_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features

@dataclass
class EvalCollator(DataCollatorWithPadding):
    max_q_len: int = 32
    max_p_len: int = 128
    def __call__(self, features):
        qry_text_ids = [x[0] for x in features]
        qry_text_features = [x[1] for x in features]
        psg_text_ids = [x[2] for x in features]
        psg_text_features = [x[3] for x in features]
        rels = [x[4] for x in features]
        if isinstance(qry_text_features[0], list):
            qry_text_features = sum(qry_text_features, [])
        if isinstance(psg_text_features[0], list):
            psg_text_features = sum(psg_text_features, [])

        qry_collated_features = self.tokenizer.pad(
            qry_text_features,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        psg_collated_features = self.tokenizer.pad(
            psg_text_features,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        return qry_text_ids, qry_collated_features, psg_text_ids, psg_collated_features, rels