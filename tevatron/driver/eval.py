import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from tevatron.data import EvalDataset, EvalCollator
from tevatron.datasets import HFEvalDataset
from tevatron.utils import metrics
METRICS_MAP = ['MAP', 'RPrec', 'NDCG', 'MRR', 'MRR@10']
# from tevatron.densification.utils import densify

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        output_hidden_states=True, 
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    if (model_args.model).lower() == 'colbert':
        from tevatron.ColBERT.modeling import ColBERTForInference
        from tevatron.ColBERT.modeling import ColBERTOutput as Output
        logger.info("Evaluating model ColBERT")
        model = ColBERTForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dhr':
        from tevatron.DHR.modeling import DHRModelForInference
        from tevatron.DHR.modeling import DHROutput as Output
        logger.info("Evaluating model DHR")
        model = DHRModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dlr':
        from tevatron.DHR.modeling import DHRModelForInference
        from tevatron.DHR.modeling import DHROutput as Output
        logger.info("Evaluating model DHR")
        model_args.combine_cls = False
        model = DHRModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'agg':
        from tevatron.Aggretriever.modeling import DenseModelForInference
        from tevatron.Aggretriever.modeling import DenseOutput as Output
        logger.info("Evaluating model Dense (AGG)")
        model = DHRModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dense':
        from tevatron.Dense.modeling import DenseModelForInference
        from tevatron.Dense.modeling import DenseOutput as Output
        logger.info("Evaluating model Dense (CLS)")
        model = DenseModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError('input model is not supported')

    eval_dataset = HFEvalDataset(tokenizer=tokenizer, data_args=data_args,
                                     cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    eval_dataset = EvalDataset(data_args, eval_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                   tokenizer)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EvalCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    model = model.to(training_args.device)
    model.eval()

    num_candidates_per_qry = 1000 
    if num_candidates_per_qry%training_args.per_device_eval_batch_size!=0:
        raise ValueError('Batch size should be a factor of {}'.format(num_candidates_per_qry))
    all_metrics = np.zeros(len(METRICS_MAP))
    num_examples = 0
    qids = []
    candidiate_psg_ids = []
    scores = []
    labels = []
    for (batch_qry_ids, batch_qry_featutres, batch_psg_ids, batch_psg_features, rels) in tqdm(eval_loader):
        if len(set(batch_qry_ids)) != 1:
            raise ValueError('Tere is other query in the Eval batch!')
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch_qry_featutres.items():
                    batch_qry_featutres[k] = v.to(training_args.device)
                for k, v in batch_psg_features.items():
                    batch_psg_features[k] = v.to(training_args.device)
                model_output: Output = model(query=batch_qry_featutres, passage=batch_psg_features)
        
        qids += batch_qry_ids
        candidiate_psg_ids += batch_psg_ids 
        scores += model_output.scores.cpu().numpy().tolist()
        labels += rels
        if len(candidiate_psg_ids) == num_candidates_per_qry:
            if len(set(qids)) != 1:
                raise ValueError('Tere is other query in the set!')
            gt = set(list(np.where(np.array(labels) > 0)[0]))

            predict_doc = np.array(scores).argsort()[::-1]
            all_metrics += metrics.metrics(gt=gt, pred=predict_doc, metrics_map=METRICS_MAP)
            num_examples+=1
            qids = []
            candidiate_psg_ids = []
            scores = []
            labels = []
            if (num_examples%10==0):
                logging.warn("Read {} examples, Metrics so far:".format(num_examples))
                logging.warn("  ".join(METRICS_MAP))
                logging.warn(all_metrics / num_examples)
                if num_examples==200:
                    break
        # Write results


    output_dir = '/'.join( (data_args.encoded_save_path).split('/')[:-1] ) 



if __name__ == "__main__":
    main()
