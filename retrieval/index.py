import argparse
import os
import glob
# os.environ['OMP_NUM_THREADS'] = str(32)
import numpy as np
import math
from progressbar import *
# from util import load_tfrecords_and_index, read_id_dict, faiss_index
from multiprocessing import Pool, Manager
import pickle
import torch
import torch.nn as nn
import time




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--index_prefix", type=str, default='msmarco-passage')
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--index_path", type=str, required=True)
	args = parser.parse_args()


	## Merge index
	corpus_files = glob.glob(os.path.join(args.index_path, args.index_prefix + '.split*.pt'))

	corpus_embs = []
	corpus_arg_idxs = []
	docids = []
	for corpus_file in corpus_files:
		with open(corpus_file, 'rb') as f:
			print('Load index: {}...'.format(corpus_file))
			corpus_emb, corpus_arg_idx, docid=pickle.load(f)
			corpus_embs.append(corpus_emb)
			corpus_arg_idxs.append(corpus_arg_idx)
			docids += docid

	print('Merge index ...')
	try:
		corpus_arg_idxs = np.concatenate(corpus_arg_idxs, axis=0)
	except:
		corpus_arg_idxs = 0
	corpus_embs = np.concatenate(corpus_embs, axis=0)

	with open(os.path.join(args.index_path, args.index_prefix + '.index.pt'), 'wb') as f:
		pickle.dump([corpus_embs, corpus_arg_idxs, docids], f, protocol=4)

	


if __name__ == "__main__":
	main()