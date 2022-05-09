import logging
import pickle
import glob
import numpy as np
import json
import argparse
from pyserini.index import IndexReader
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

def vectorize_and_densify(files, dim, omission_num, output_path):
    data_num = 0
    total_collision_num = 0
    for file in files:
        f = open(file, 'r')
        print('count line number')
        for line in f:
            data_num+=1
        f.close()
        print('initialize numpy array with {}X{}'.format(data_num, dim))
        value_encoded = np.zeros((data_num, dim), dtype=np.float16)
        index_encoded = np.zeros((data_num, dim), dtype=np.int16)
        docids =[]
        with open(file, 'r') as f:
            for i, line in tqdm(enumerate(f), total=data_num, desc=f"densify {file}"):
                data = json.loads(line)
                docids.append(data['id'])
                value, index, collision_num = densify(data, dim, omission_num)
                total_collision_num += collision_num
                value_encoded[i] = value
                index_encoded[i] = index
    print('Total {} collisions with {} passages'.format(total_collision_num, data_num))
    with open(output_path, 'wb') as f_out:
        pickle.dump([value_encoded, index_encoded, docids], f_out, protocol=4)

def main():
    parser = argparse.ArgumentParser(
        description='Transform corpus into wordpiece corpus')
    parser.add_argument('--index_path', required=True, help='anserini index path')
    parser.add_argument('--vector_dir', required=True, help='directory with json files')
    parser.add_argument('--output_dir', required=True, help='output pickle path')
    parser.add_argument('--output_dims', type=int, required=False, default=768)
    parser.add_argument('--num_workers', type=int, required=False, default=None)
    parser.add_argument('--prefix', required=True, help='index name prefix')
    args = parser.parse_args()

    index_reader = IndexReader(args.index_path)
    total_num_vocabs = index_reader.stats()['unique_terms']  
    omission_num = total_num_vocabs%args.output_dims

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    densified_vector_dir  = os.path.join(args.output_dir, f'encoding')
    if not os.path.exists(densified_vector_dir):
        os.mkdir(densified_vector_dir)

    files = glob.glob(os.path.join(args.vector_dir, '*.json'))

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
            pool.apply_async(vectorize_and_densify ,(files[start:], args.output_dims, omission_num, output_path))
        else:
            pool.apply_async(vectorize_and_densify ,(files[start:(start+num_files_per_worker)], args.output_dims, omission_num, output_path))

    pool.close()  
    pool.join()  

    

if __name__ == '__main__':
    main()    
