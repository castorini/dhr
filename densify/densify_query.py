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
logger = logging.getLogger(__name__)

def densify(data, dim, omission_num):
    value = np.zeros((dim), dtype=np.float16)
    index = np.zeros((dim), dtype=np.int16)
    collision_num = 0
    for i, (token_id, weight) in enumerate(zip(data['token_id'],data['weights'])):
        if token_id < omission_num:
            continue
        else:
            slice_num = (token_id - omission_num)%dim
            index_num = (token_id - omission_num)//dim
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

    

def main():
    parser = argparse.ArgumentParser(
        description='Transform corpus into wordpiece corpus')
    parser.add_argument('--index_path', required=True, help='anserini index path')
    parser.add_argument('--query_path', required=True, help='directory with json files')
    parser.add_argument('--output_dims', type=int, required=False, default=768)
    parser.add_argument('--output_dir', required=True, help='output pickle path')
    parser.add_argument('--prefix', required=True, help='index name prefix')
    parser.add_argument('--analyze' , action='store_true', help='turn on when densifying BM25')
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir  = os.path.join(args.output_dir, f'encoding')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    densified_vector_dir = os.path.join(args.output_dir, f"queries")
    if not os.path.exists(densified_vector_dir):
        os.mkdir(densified_vector_dir)
    
    analyzer = Analyzer(get_lucene_analyzer())
    index_reader = IndexReader(args.index_path)
    total_num_vocabs = index_reader.stats()['unique_terms']  
    
    term2idx = {}
    for idx, term in tqdm(enumerate(index_reader.terms()), desc=f"read index terms"): 
        term2idx[term.term] = idx

    omission_num = total_num_vocabs%args.output_dims

    

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
    total_collision_num = 0
    with open(args.query_path, 'r') as f:
        for i, line in enumerate(f):
            qid, query = line.strip().split('\t')
            if args.analyze:
                analyzed_query_terms = analyzer.analyze(query)
            else:
                analyzed_query_terms = query.split(' ')
            
            id2weight = defaultdict(int)
            for analyzed_query_term in analyzed_query_terms:
                try:
                    term_idx = term2idx[analyzed_query_term]
                    id2weight[term_idx] += 1
                except:
                    continue

            data = {'token_id':[], 'weights':[]}
            for (token_id, weight) in id2weight.items():
                data['token_id'].append(token_id)
                data['weights'].append(weight)
            
            qids.append(qid)
            value, index, collision_num = densify(data, args.output_dims, omission_num)
            total_collision_num += collision_num
            value_encoded[i] = value
            index_encoded[i] = index  

            # ###########for debug###############
            # text = f"Form view is the primary means of adding and modifying data in tables. \
            # You can also change the design of a form in this view. format . \
            # Specifies how data is displayed and printed. An Access database provides standard formats for specific data types, \
            # as does an Access project for the equivalent SQL data types. You can also create custom formats." 
            # analyzed_query_terms = analyzer.analyze(text)
            
            # id2weight = defaultdict(int)
            # for analyzed_query_term in analyzed_query_terms:
            #     try:
            #         term_idx = term2idx[analyzed_query_term]
            #         id2weight[term_idx] += 1
            #     except:
            #         continue
            # data = {'token_id':[], 'weights':[]}
            # for (token_id, weight) in id2weight.items():
            #     data['token_id'].append(token_id)
            #     data['weights'].append(weight)
            # value, index, collision_num = densify(data, args.output_dims, omission_num)

    print('Total {} collisions with {} queries'.format(total_collision_num, i+1))
    file_name = args.prefix + '.' + (args.query_path).split('/')[-1].replace('tsv','pt')
    output_path = os.path.join(densified_vector_dir, file_name)
    with open(output_path, 'wb') as f_out:
        pickle.dump([value_encoded, index_encoded, qids], f_out, protocol=4)

    

if __name__ == '__main__':
    main()    
