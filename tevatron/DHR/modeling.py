import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, PreTrainedModel, AutoModelForMaskedLM
from transformers.modeling_outputs import ModelOutput


from typing import Optional, Dict

from ..arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


@dataclass
class DHROutput(ModelOutput):
    q_semantic_reps: Tensor = None
    q_lexical_reps: Tensor = None
    p_semantic_reps: Tensor = None
    p_lexical_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True, 
            name='pooler'
    ):
        super(LinearPooler, self).__init__()
        self.name = name
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, '{}.pt'.format(self.name))
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, '{}.pt'.format(self.name)), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training {} from scratch".format(self.name))
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, '{}.pt'.format(self.name)))
        with open(os.path.join(save_path, '{}_config.json').format(self.name), 'w') as f:
            json.dump(self._config, f)


class DHRModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            term_weight_trans: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.term_weight_trans = term_weight_trans

        self.softmax = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Todo: 
        if model_args.combine_cls:
            self.lamb = 1
        else:
            self.lamb = 0
        self.temperature = 1

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            teacher_scores: Tensor = None,
    ):

        

        q_lexical_reps, q_semantic_reps = self.encode_query(query)
        p_lexical_reps, p_semantic_reps = self.encode_passage(passage)

        if q_lexical_reps is None or p_lexical_reps is None:
            return DHROutput(
                q_lexical_reps = q_lexical_reps,
                q_semantic_reps = q_semantic_reps,
                p_lexical_reps = p_lexical_reps,
                p_semantic_reps = p_semantic_reps,
            )

        if self.training:
            if self.train_args.negatives_x_device:
                q_lexical_reps = self.dist_gather_tensor(q_lexical_reps)
                p_lexical_reps = self.dist_gather_tensor(p_lexical_reps)
                q_semantic_reps = self.dist_gather_tensor(q_semantic_reps)
                p_semantic_reps = self.dist_gather_tensor(p_semantic_reps)
                if teacher_scores is not None:
                    teacher_scores = self.dist_gather_tensor(teacher_scores)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            
            # todo: add tct loss
            if self.model_args.kd:
                if teacher_scores is None:
                    raise ValueError(f"No pairwise teacher score for knowledge distillation!")
                q_lexical_reps = q_lexical_reps.view(effective_bsz, 1, -1)
                p_lexical_reps = p_lexical_reps.view(effective_bsz, 2, -1)
                pair_lexical_scores = torch.matmul(q_lexical_reps, p_lexical_reps.transpose(2, 1)).squeeze()

                q_semantic_reps = q_semantic_reps.view(effective_bsz, 1, -1)
                p_semantic_reps = p_semantic_reps.view(effective_bsz, 2, -1)
                pair_semantic_scores = torch.matmul(q_semantic_reps, p_semantic_reps.transpose(2, 1)).squeeze()

                scores = pair_lexical_scores + lamb * pair_semantic_scores

                loss = self.kl_loss(nn.functional.log_softmax(scores, dim=-1), self.softmax(teacher_scores))
            else:
                # lexical matching
                lexical_scores = torch.matmul(q_lexical_reps, p_lexical_reps.transpose(0, 1))
                lexical_scores = lexical_scores.view(effective_bsz, -1)

                # semantic matching
                semantic_scores = torch.matmul(q_semantic_reps, p_semantic_reps.transpose(0, 1))
                semantic_scores = semantic_scores.view(effective_bsz, -1)

                # score fusion
                scores = lexical_scores + self.lamb * semantic_scores

                target = torch.arange(
                    scores.size(0),
                    device=scores.device,
                    dtype=torch.long
                )
                target = target * self.data_args.train_n_passages

                loss = self.cross_entropy(scores * temperature, target)



            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return DHROutput(
                loss=loss,
                scores=scores,
                q_lexical_reps=q_lexical_reps,
                q_semantic_reps=q_semantic_reps,
                p_lexical_reps=p_lexical_reps,
                p_semantic_reps=p_semantic_reps,
            )

        else:
            loss = None
            if query and passage:
                # lexical matching
                lexical_scores = (q_lexical_reps * p_lexical_reps).sum(1)

                # semantic matching
                semantic_scores = (q_semantic_reps * p_semantic_reps).sum(1)

                # score fusion
                scores = lexical_scores + self.lamb * semantic_scores # lambda=1
            else:
                scores = None

            return DHROutput(
                loss=loss,
                scores=scores,
                q_lexical_reps = q_lexical_reps,
                q_semantic_reps = q_semantic_reps,
                p_lexical_reps = p_lexical_reps,
                p_semantic_reps = p_semantic_reps,
            )

    def encode_passage(self, psg):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=True)
        p_seq_hidden = psg_out.hidden_states[-1]
        p_cls_hidden = p_seq_hidden[:,0] # get [CLS] embeddings
        p_logits = psg_out.logits[:,1:] # batch, seq, vocab

        # Here we slightly modify the orginal SPLADEMAX, turning into dense vectors
        p_term_weights = self.term_weight_trans(p_seq_hidden[:,1:]) # batch, seq, 1
        p_logits = self.softmax(p_logits)
        attention_mask = psg['attention_mask'][:,1:].unsqueeze(-1)
        p_lexical_reps = torch.max((p_logits * p_term_weights) * attention_mask, dim=-2).values

        
        ## This is for uniCOIL
        # p_full_term_weights = torch.zeros(p_logits.shape[0], p_logits.shape[1], p_logits.shape[2], dtype=torch.float16, device=p_logits.device) # (batch, seq, vocab)
        # p_full_term_weights = torch.scatter(p_full_term_weights, dim=-1, index=psg.input_ids[:,1:,None], src=p_term_weights) # (batch, seq, vocab)

        ## Original SPLADEMax
        # attention_mask = psg['attention_mask'][:, 1:].unsqueeze(-1)
        # p_lexical_reps = torch.max(torch.log(1 + torch.relu(p_logits)) * attention_mask, dim=1).values
        
        if self.pooler is not None:
            p_semantic_reps = self.pooler(p=p_cls_hidden)  # D * d
        else:
            p_semantic_reps = p_cls_hidden
        return p_lexical_reps, p_semantic_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None

        qry_out = self.lm_q(**qry, return_dict=True)
        q_seq_hidden = qry_out.hidden_states[-1] 
        q_cls_hidden = q_seq_hidden[:,0] # get [CLS] embeddings
        q_logits = qry_out.logits[:,1:] # batch, seq-1, vocab
        
        # Here we slightly modify the orginal SPLADEMAX, turning into dense vectors
        q_term_weights = self.term_weight_trans(q_seq_hidden[:,1:]) # batch, seq, 1
        q_logits = self.softmax(q_logits)
        attention_mask = qry['attention_mask'][:,1:].unsqueeze(-1)
        q_lexical_reps = torch.sum((q_logits * q_term_weights) * attention_mask, dim=-2)
        
        ## This is for uniCOIL
        # q_full_term_weights = torch.zeros(q_logits.shape[0], q_logits.shape[1], q_logits.shape[2], dtype=torch.float16, device=q_logits.device) # (batch, len, vocab)
        # q_full_term_weights = torch.scatter(q_full_term_weights, dim=-1, index=qry.input_ids[:,1:,None], src=q_term_weights) # fill value
        
        ## Original SPLADEMax
        # attention_mask = qry['attention_mask'][:, 1:].unsqueeze(-1)
        # q_lexical_reps = torch.max(torch.log(1 + torch.relu(q_logits)) * attention_mask, dim=1).values
        
        if self.pooler is not None:
            q_semantic_reps = self.pooler(q=q_cls_hidden)
        else:
            q_semantic_reps = q_cls_hidden
        return q_lexical_reps, q_semantic_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def build_term_weight_transform(model_args):
        term_weight_trans = LinearPooler(
            model_args.projection_in_dim,
            1,
            tied=not model_args.untie_encoder,
            name="TermWeightTrans"
        )
        term_weight_trans.load(model_args.model_name_or_path)
        return term_weight_trans

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModelForMaskedLM.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModelForMaskedLM.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        term_weight_trans = cls.build_term_weight_transform(model_args)

        # Todo: Freeze embedding layer
        for name, param in lm_q.named_parameters():
            if 'word_embeddings' in name:
                param.requires_grad = False


        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            term_weight_trans=term_weight_trans,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)
        self.term_weight_trans.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

class DHRModelForInference(DHRModel):
    POOLER = LinearPooler  

    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            term_weight_trans: nn.Module = None,
            lamb = 1,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.term_weight_trans = term_weight_trans
        self.softmax = nn.Softmax(dim=-1)
        self.lamb = lamb

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(DHRModelForInference, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(DHRModelForInference, self).encode_query(qry)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModelForMaskedLM.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModelForMaskedLM.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        TermWeightTrans_weights = os.path.join(model_name_or_path, 'TermWeightTrans.pt')
        TermWeightTrans_config = os.path.join(model_name_or_path, 'TermWeightTrans_config.json')
        if os.path.exists(TermWeightTrans_weights) and os.path.exists(TermWeightTrans_config):
            logger.info(f'found TermWeightTrans weight and configuration')
            with open(TermWeightTrans_config) as f:
                TermWeightTrans_config_dict = json.load(f)
            # Todo: add name to config
            TermWeightTrans_config_dict['name'] = 'TermWeightTrans'
            term_weight_trans = cls.POOLER(**TermWeightTrans_config_dict)
            term_weight_trans.load(model_name_or_path)
        else:
            term_weight_trans = None
        
        if model_args.combine_cls:
            lamb = 1
        else:
            lamb = 0

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            term_weight_trans=term_weight_trans,
            lamb=lamb
        )
        return model