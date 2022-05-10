import logging
import pickle
import glob
import numpy as np
import gzip
import json
import argparse
from pyserini.index import IndexReader
from multiprocessing import Pool, Manager, Queue
from transformers import AutoModelForMaskedLM, AutoTokenizer
import multiprocessing
import os
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

omission_num = \
    {'bm25': 472,
     'deepimpact': 502,
     'unicoil': 570, 
     'splade': 570}

whole_word_matching = \
    {'bm25': True,
     'deepimpact': True,
     'unicoil': False, 
     'splade': False}

def densify(data, dim, whole_word_matching, token2id, args):
    value = np.zeros((dim), dtype=np.float16)
    if whole_word_matching:
        index = np.zeros((dim), dtype=np.int16)
    else:
        index = np.zeros((dim), dtype=np.int8)
    collision_num = 0
    for i, (token, weight) in enumerate(data['vector'].items()):
        token_id = token2id[token]
        if token_id < omission_num[args.model]:
            continue
        else:
            slice_num = (token_id - omission_num[args.model])%dim
            index_num = (token_id - omission_num[args.model])//dim
            if value[slice_num]==0:
                value[slice_num] = weight
                index[slice_num] = index_num
            else:
                # collision
                collision_num += 1
                if value[slice_num] < weight:
                    value[slice_num] = weight
                    index[slice_num] = index_num
    return value, index, collision_num


def vectorize_and_densify(files, file_type, dim, whole_word_matching, token2id, output_path, args):
    data_num = 0
    logger.info('count line number')
    for file in files:
        if file_type == 'jsonl.gz':
            f = gzip.open(file, "rb")
        else:
            f = open(file, 'r')
        for line in f:
            data_num+=1
        f.close()

    logger.info('initialize numpy array with {}X{}'.format(data_num, dim))
    value_encoded = np.zeros((data_num, dim), dtype=np.float16)
    if whole_word_matching:
        index_encoded = np.zeros((data_num, dim), dtype=np.int16)
    else:
        index_encoded = np.zeros((data_num, dim), dtype=np.int8)
    docids =[]
    total_collision_num = 0
    counter = 0
    for file in files:
        if file_type == 'jsonl.gz':
            f = gzip.open(file, "rb")
        else:
            f = open(file, 'r')
        for i, line in tqdm(enumerate(f), desc=f"densify {file}"):
            data = json.loads(line)
            docids.append(data['id'])
            value, index, collision_num = densify(data, dim, whole_word_matching, token2id, args)
            total_collision_num += collision_num
            value_encoded[counter] = value
            index_encoded[counter] = index
            counter += 1
        f.close()

    print('Total {} collisions with {} passages'.format(total_collision_num, data_num))
    with open(output_path, 'wb') as f_out:
        pickle.dump([value_encoded, index_encoded, docids], f_out, protocol=4)


def get_files(directory):
    files = glob.glob(os.path.join(directory, '*.json'))
    if len(files) == 0:
        files = glob.glob(os.path.join(directory, '*.jsonl.gz'))
        file_type = 'jsonl.gz'
    else:
        file_type = 'json'
    if len(files) == 0:
        raise ValueError('There is no json or jsonl.gz files in {}'.format(directory))
    return files, file_type

def main():
    parser = argparse.ArgumentParser(description='Densify corpus')
    parser.add_argument('--model', required=True, help='bm25, deepimpact, unicoil or splade')
    parser.add_argument('--tokenizer', required=False, default="bert-base-uncased", help='anserini index path or transformer tokenizer')
    parser.add_argument('--vector_dir', required=True, help='directory with json files')
    parser.add_argument('--output_dir', required=True, help='output pickle directory')
    parser.add_argument('--output_dims', type=int, required=False, default=768)
    parser.add_argument('--num_workers', type=int, required=False, default=None)
    parser.add_argument('--prefix', required=True, help='index name prefix')
    args = parser.parse_args()

    token2id = {}
    if (args.model == 'bm25') or (args.model == 'deepimpact'):
        tokenizer = IndexReader(args.tokenizer)
        for idx, token in tqdm(enumerate(tokenizer.terms()), desc=f"read index terms"): 
            token2id[token.term] = idx
    elif (args.model == 'unicoil') or (args.model == 'splade'):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        token2id = tokenizer.vocab
    else:
        raise ValueError('We cannot handle you input model')

    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    densified_vector_dir  = os.path.join(args.output_dir, f'encoding')
    if not os.path.exists(densified_vector_dir):
        os.mkdir(densified_vector_dir)

    files, file_type = get_files(args.vector_dir)

    total_num_files = len(files)
    if args.num_workers is None:
        args.num_workers = total_num_files
        num_files_per_worker = 1
    else:
        num_files_per_worker = total_num_files//args.num_workers
        if (total_num_files%args.num_workers) != 0:
            args.num_workers+=1

    pool = Pool(args.num_workers)
    for i in range(args.num_workers):
        start = i*num_files_per_worker
        output_path = os.path.join(densified_vector_dir, f"{args.prefix}.split{i}.pt")

        if i==(args.num_workers-1):
            pool.apply_async(vectorize_and_densify ,(files[start:], file_type, args.output_dims, whole_word_matching[args.model], token2id, output_path, args))
        else:
            pool.apply_async(vectorize_and_densify ,(files[start:(start+num_files_per_worker)], file_type, args.output_dims, whole_word_matching[args.model], token2id, output_path, args))

        # for debug
        # vectorize_and_densify(files[start:(start+num_files_per_worker)], file_type, args.output_dims, whole_word_matching[args.model], token2id, output_path, args)

    pool.close()  
    pool.join()  

    

if __name__ == '__main__':
    main()    
