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
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.datasets import HFQueryDataset, HFCorpusDataset
from tevatron.DHR.utils import densify

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
    
    if (model_args.model).lower() == 'dhr':
        from tevatron.DHR.modeling import DHRModelForInference
        from tevatron.DHR.modeling import DHROutput as Output
        logger.info("Encoding model DHR")
        model = DHRModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dlr':
        from tevatron.DHR.modeling import DHRModelForInference
        from tevatron.DHR.modeling import DHROutput as Output
        logger.info("Encoding model DLR")
        model_args.combine_cls = False
        model = DHRModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'agg':
        from tevatron.Aggretriever.modeling import DenseModelForInference
        from tevatron.Aggretriever.modeling import DenseOutput as Output
        logger.info("Encoding model Dense (AGG)")
        model = DenseModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dense':
        from tevatron.Dense.modeling import DenseModelForInference
        from tevatron.Dense.modeling import DenseOutput as Output
        logger.info("Encding model Dense (CLS)")
        model = DenseModelForInference.build(
            model_args=model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError('input model is not supported')

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                        cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                   tokenizer, max_len=text_max_length)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    

    def initialize_reps(data_num, dim, dtype):
        return np.zeros((data_num, dim), dtype=dtype)


    offset = 0
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    data_num = len(encode_dataset)
    value_encoded, index_encoded = None, None

    for (batch_ids, batch) in tqdm(encode_loader):
        batch_size = len(batch_ids)
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)

                if data_args.encode_is_qry:
                    if (model_args.model).lower() == 'agg':
                        model_output: Output = model(query=batch, agg_dim=model_args.agg_dim)
                    else:
                        model_output: Output = model(query=batch)

                    if (model_args.model).lower() == 'dense' or (model_args.model).lower() == 'agg':
                        reps = model_output.q_reps.cpu().detach().numpy()
                        if value_encoded is None:
                            value_encoded = initialize_reps(data_num, reps.shape[1], np.float16)
                        value_encoded[offset: (offset + batch_size), :] = reps
                    else:
                        dlr_value_reps, dlr_index_reps = densify(model_output.q_lexical_reps, model_args.dlr_out_dim)
                        dlr_value_reps = dlr_value_reps.cpu().detach().numpy()
                        dlr_index_reps = dlr_index_reps.cpu().detach().numpy().astype(np.uint8)
                        cls_reps = model_output.q_semantic_reps.cpu().detach().numpy()

                        if value_encoded is None:
                            if cls_reps is None:
                                cls_dim = 0
                            else:
                                cls_dim = cls_reps.shape[1]
                            value_encoded = initialize_reps(data_num, dlr_value_reps.shape[1] + cls_dim, np.float16)
                            index_encoded = initialize_reps(data_num, dlr_index_reps.shape[1], np.uint8)
                        value_encoded[offset: (offset + batch_size), :model_args.dlr_out_dim] = dlr_value_reps
                        index_encoded[offset: (offset + batch_size), :model_args.dlr_out_dim] = dlr_index_reps
                        if cls_reps is not None:
                            value_encoded[offset: (offset + batch_size), model_args.dlr_out_dim:] = cls_reps

                else:
                    if (model_args.model).lower() == 'agg':
                        model_output: Output = model(passage=batch, agg_dim=model_args.agg_dim)
                    else:
                        model_output: Output = model(passage=batch)
                    if (model_args.model).lower() == 'dense' or (model_args.model).lower() == 'agg':
                        reps = model_output.p_reps.cpu().detach().numpy()
                        if value_encoded is None:
                            value_encoded = initialize_reps(data_num, reps.shape[1], np.float16)
                        value_encoded[offset: (offset + batch_size), :] = reps
                    else:
                        dlr_value_reps, dlr_index_reps = densify(model_output.p_lexical_reps, model_args.dlr_out_dim)
                        dlr_value_reps = dlr_value_reps.cpu().detach().numpy()
                        dlr_index_reps = dlr_index_reps.cpu().detach().numpy().astype(np.uint8)
                        cls_reps = model_output.p_semantic_reps.cpu().detach().numpy()

                        if value_encoded is None:
                            if cls_reps is None:
                                cls_dim = 0
                            else:
                                cls_dim = cls_reps.shape[1]
                            value_encoded = initialize_reps(data_num, dlr_value_reps.shape[1] + cls_dim, np.float16)
                            index_encoded = initialize_reps(data_num, dlr_index_reps.shape[1], np.uint8)
                        value_encoded[offset: (offset + batch_size), :model_args.dlr_out_dim] = dlr_value_reps
                        index_encoded[offset: (offset + batch_size), :model_args.dlr_out_dim] = dlr_index_reps
                        if cls_reps is not None:
                            value_encoded[offset: (offset + batch_size), model_args.dlr_out_dim:] = cls_reps

        offset += batch_size

    output_dir = '/'.join( (data_args.encoded_save_path).split('/')[:-1] ) 
    if not os.path.exists(output_dir):
        logger.info(f'{output_dir} not exists, create')
        os.mkdir(output_dir)
    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump([value_encoded, index_encoded, lookup_indices], f, protocol=4)


if __name__ == "__main__":
    main()
