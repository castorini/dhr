import logging
import pickle
import glob
import numpy as np
import json
import argparse
from collections import defaultdict
from pyserini.index import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
from multiprocessing import Pool, Manager, Queue
import multiprocessing
import os
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini.encode import QueryEncoder, TokFreqQueryEncoder, UniCoilQueryEncoder
from .densify_corpus import densify, whole_word_matching


def main():
    parser = argparse.ArgumentParser(
        description='Transform corpus into wordpiece corpus')
    parser.add_argument('--model', required=True, help='bm25, deepimpact, unicoil or splade')
    parser.add_argument('--tokenizer', required=False, default="bert-base-uncased", help='anserini index path or transformer tokenizer')
    parser.add_argument('--query_path', required=True, help='query tsv file')
    parser.add_argument('--output_dims', type=int, required=False, default=768)
    parser.add_argument('--output_dir', required=True, help='output pickle directory')
    parser.add_argument('--prefix', required=True, help='index name prefix')
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir  = os.path.join(args.output_dir, f'encoding')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    densified_vector_dir = os.path.join(args.output_dir, f"queries")
    if not os.path.exists(densified_vector_dir):
        os.mkdir(densified_vector_dir)
    
    
    args.model = args.model.lower()
    token2id = {}
    if (args.model == 'bm25') or (args.model == 'deepimpact'):
        analyzer = Analyzer(get_lucene_analyzer())
        tokenizer = IndexReader(args.tokenizer)
        for idx, token in tqdm(enumerate(tokenizer.terms()), desc=f"read index terms"): 
            token2id[token.term] = idx
        if args.model == 'bm25':
            analyze = True
        else:
            analyze = False
        query_encoder = None        
    elif (args.model == 'unicoil') or (args.model == 'splade'):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        token2id = tokenizer.vocab
        if args.model == 'unicoil':
            query_encoder = UniCoilQueryEncoder('castorini/unicoil-msmarco-passage')
    else:
        raise ValueError('We cannot handle you input --model')


    f = open(args.query_path, 'r')
    print('count line number')
    data_num = 0
    for line in f:
        data_num+=1
    f.close()

    print('initialize numpy array with {}X{}'.format(data_num, args.output_dims))
    value_encoded = np.zeros((data_num, args.output_dims), dtype=np.float16)
    index_encoded = np.zeros((data_num, args.output_dims), dtype=np.int16)

    qids = []
    data = {}
    total_collision_num = 0
    with open(args.query_path, 'r') as f:
        for i, line in enumerate(f):
            qid, query = line.strip().split('\t')
            if query_encoder is None:
                if analyze:
                    analyzed_query_terms = analyzer.analyze(query)
                else:
                    analyzed_query_terms = query.split(' ')
                # use tf as term weight
                vector = defaultdict(int)
                for analyzed_query_term in analyzed_query_terms:
                    vector[analyzed_query_term] += 1
            else:
                vector = query_encoder.encode(query)

            data['vector'] = vector
            
            qids.append(qid)
            value, index, collision_num = densify(data, args.output_dims, whole_word_matching[args.model] , token2id, args)
            total_collision_num += collision_num
            value_encoded[i] = value
            index_encoded[i] = index  



    print('Total {} collisions with {} queries'.format(total_collision_num, i+1))
    file_name = args.prefix + '.' + (args.query_path).split('/')[-1].replace('tsv','pt')
    output_path = os.path.join(densified_vector_dir, file_name)
    with open(output_path, 'wb') as f_out:
        pickle.dump([value_encoded, index_encoded, qids], f_out, protocol=4)

    

if __name__ == '__main__':
    main()    
