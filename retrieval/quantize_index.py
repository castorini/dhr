import argparse
import os
import numpy as np
import pickle5 as pickle
import faiss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--output_index_path", type=str, default=None)
    parser.add_argument("--qauntized_dim", type=int, default=64)
    parser.add_argument("--n_bits", type=int, default=8)
    args = parser.parse_args()

    if args.output_index_path is None:
        # assign to index dir
        index_dir = '/'.join(index_path.split('/')[:-1])
        args.output_index_path = os.path.join(index_dir, 'pq{}_index'.format(args.qauntized_dim))

    # load index
    print('Load index ...')
    with open(args.index_path, 'rb') as f:
        corpus_embs, corpus_arg_idxs, docids=pickle.load(f)
        corpus_embs = corpus_embs.astype(np.float32)

    faiss.omp_set_num_threads(36)
    print('build PQ index...')
    index = faiss.IndexPQ(corpus_embs.shape[1], args.qauntized_dim, args.n_bits, faiss.METRIC_INNER_PRODUCT)
    index.verbose = True

    print('train PQ...')
    index.train(corpus_embs)
    print('build index...')
    index.add(corpus_embs)
    print('write index to {}'.format(args.output_index_path))
    faiss.write_index(index, args.output_index_path)
    print('finish')


if __name__ == "__main__":
	main()
