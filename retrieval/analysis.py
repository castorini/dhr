import argparse
import os
import glob
# os.environ['OMP_NUM_THREADS'] = str(32)
import numpy as np
import math
from progressbar import *
from util import load_tfrecords_and_index, read_id_dict, faiss_index
from multiprocessing import Pool, Manager
import pickle
import torch
import torch.nn as nn
torch.cuda.set_device(0)
import time
import faiss
from collections import defaultdict


def reconstruct_bow(query_embs, query_arg_idxs, id2vocab, vocab2id, dimensions, threshold):
	reconstruct_vocab = defaultdict(list)

	for i, query_emb in enumerate(query_embs):
		dimension = dimensions[i]
		query_arg_idx = query_arg_idxs[i]
		outputs = query_emb>threshold
		stride = 29952/dimension
		for j, output in enumerate(outputs):
			if output:
				index = query_arg_idx[j]
				vocab_id = 570 + j + index*dimension
				reconstruct_vocab[i].append(id2vocab[vocab_id])
	print(' '.join(reconstruct_vocab[1]))
	loss_vocab = set(reconstruct_vocab[1]) - set(reconstruct_vocab[0])

	for vocab in loss_vocab:
		vocab_id = vocab2id[vocab]
		for collision_vocab in reconstruct_vocab[0]:
			collision_vocab_id = vocab2id[collision_vocab]
			if ((collision_vocab_id - vocab_id)%dimensions[0])==0:
				print('{} replace {}'.format(collision_vocab, vocab))





def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--query_emb_path0", type=str, required=True, help='The embedding file or the dir to save all the files with .tf')
	parser.add_argument("--query_emb_path1", type=str, required=True, help='The embedding file or the dir to save all the files with .tf')
	parser.add_argument("--doc_word_num", type=int, default=1, help='in case when using token embedding maxsim search instead of pooling embedding')
	parser.add_argument("--emb_dim0", type=int, default=768)
	parser.add_argument("--emb_dim1", type=int, default=768)
	parser.add_argument("--passages_per_file", type=int, default=1000000, help='our default tf record include 1000,000 passages per file')
	parser.add_argument("--data_type", type=str, default='16', help='16 or 32 bit')
	parser.add_argument("--add_cls", action='store_true')
	parser.add_argument("--id_to_doc_path", type=str, default=None)
	parser.add_argument("--vocab_path", type=str, required=True)
	parser.add_argument("--index_path0", type=str, required=True)
	parser.add_argument("--index_path1", type=str, required=True)
	parser.add_argument("--query", type=str)
	args = parser.parse_args()






	query_texts = []
	query_ids = []
	qid2qidx = {}
	with open(args.query, 'r') as f:
		for i, line in enumerate(f):
			query_id, query_text = line.split('\t')
			query_texts.append(query_text)
			query_ids.append(query_id)
			qid2qidx[query_id] = i
	print('Load query embeddings: {}...'.format(args.query_emb_path0))
	query_embs0, query_arg_idxs0, qids0=load_tfrecords_and_index([args.query_emb_path0],\
								data_num=800000, word_num=1, \
								dim=args.emb_dim0, data_type=args.data_type, add_cls=args.add_cls)
	print('Load query embeddings: {}...'.format(args.query_emb_path1))
	query_embs1, query_arg_idxs1, qids1=load_tfrecords_and_index([args.query_emb_path1],\
								data_num=800000, word_num=1, \
								dim=args.emb_dim1, data_type=args.data_type, add_cls=args.add_cls)

		


	id2vocab = {}
	vocab2id = {}
	with open(args.vocab_path, 'r') as f:
		for i, line in enumerate(f):
			vocab = line.strip()
			id2vocab[i] = vocab
			vocab2id[vocab] = i
	qid = '323815'
	reconstruct_bow([query_embs0[qid2qidx[qid]], query_embs1[qid2qidx[qid]]], [query_arg_idxs0[qid2qidx[qid]],query_arg_idxs1[qid2qidx[qid]]], id2vocab, vocab2id, [args.emb_dim0,args.emb_dim1], 0.1)
	# reconstruct_bow(query_embs1[qid2qidx[qid]], query_arg_idxs1[qid2qidx[qid]], id2vocab, args.emb_dim1, 0.1)

	index_file0 = os.path.join(args.index_path0,'index.pickle')
	print('Load index: {}...'.format(index_file0))
	
	with open(index_file0, 'rb') as f:
		corpus_embs0, corpus_arg_idxs0, docids=pickle.load(f)

	index_file1 = os.path.join(args.index_path1,'index.pickle')
	print('Load index: {}...'.format(index_file1))
	with open(index_file1, 'rb') as f:
		corpus_embs1, corpus_arg_idxs1, docids=pickle.load(f)
	docid2idx = {}
	for i, docid in enumerate(docids):
		docid2idx[docid]=i
	docid = docid2idx[32815]
	reconstruct_bow([corpus_embs0[docid], corpus_embs1[docid]], [corpus_arg_idxs0[docid],corpus_arg_idxs1[docid]], id2vocab, vocab2id, [args.emb_dim0,args.emb_dim1], 0.1)
	# reconstruct_bow(corpus_embs1[docid], corpus_arg_idxs1[docid], id2vocab, args.emb_dim1, 0.1)
	import pdb; pdb.set_trace()  # breakpoint 6e34c1da //


if __name__ == "__main__":
	main()