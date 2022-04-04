import argparse
import os
import glob
# os.environ['OMP_NUM_THREADS'] = str(32)
import numpy as np
import math
from progressbar import *
from util import load_jsonl_and_index, read_id_dict, faiss_index
from multiprocessing import Pool, Manager
import pickle
import torch
import torch.nn as nn
torch.cuda.set_device(0)
import time
import faiss


def write_id(id_to_doc_path, index_path, docids=[]):
	idx_to_docid, docid_to_idx = read_id_dict(id_to_doc_path)
	print('Write id to index folder...')
	fout = open(os.path.join(index_path, 'docid'), 'w')
	if len(docids)==0:
		for idx in range(len(idx_to_docid)):
			fout.write('{}\n'.format(idx_to_docid[idx]))
	else:
		for idx in docids:
			fout.write('{}\n'.format(idx_to_docid[idx]))
	fout.close()

def faiss_search(query_embs, corpus_embs, batch=1, topk=1000):
	dimension = query_embs.shape[1]
	res = faiss.StandardGpuResources()
	res.noTempMemory()
	# res.setTempMemory(1000 * 1024 * 1024) # 1G GPU memory for serving query
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.device = 0
	flat_config.useFloat16=True
	index = faiss.GpuIndexFlatIP(res, dimension, flat_config)
	# print("Load index ("+index_file+ ") to GPU...")
	index.add(corpus_embs)

	Distance = []
	Index = []
	print("Search with batch size %d"%(batch))
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	           ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=query_embs.shape[0]//batch).start()
	start_time = time.time()

	for i in range(query_embs.shape[0]//batch):
		D,I=index.search(query_embs[i*batch:(i+1)*batch], topk)


		Distance.append(D)
		Index.append(I)
		pbar.update(i + 1)


	D,I=index.search(query_embs[(i+1)*batch:], topk)


	Distance.append(D)
	Index.append(I)

	time_per_query = (time.time() - start_time)/query_embs.shape[0]
	print('Retrieving {} queries ({:0.3f} s/query)'.format(query_embs.shape[0], time_per_query))
	Distance = np.concatenate(Distance, axis=0)
	Index = np.concatenate(Index, axis=0)
	return Distance, Index

def search(query_embs, query_arg_idxs, corpus_embs, corpus_arg_idxs, all_results, all_scores, topk):
		widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
			' ', ETA(), ' ', FileTransferSpeed()]
		pbar = ProgressBar(widgets=widgets, maxval=10*len(query_embs)).start()
		print('search ...')
		partial_scores = {}
		partial_results = {}

		for i, query_emb in enumerate(query_embs):

			qidx = int(query_emb[0])
			query_emb = query_emb[1:]
			query_arg_idx = query_arg_idxs[qidx]

			num_idx = int((query_emb > 0.01).sum())
			important_idx = torch.argsort(query_emb, axis=0, descending=True)[:num_idx].tolist()
			# mapping = sparse.csr_matrix(corpus_arg_idxs[:,important_idx]==query_arg_idx[important_idx])
			# mapping =  mapping.multiply(corpus_embs[:,important_idx])
			# mapping =  mapping.multiply(query_emb[important_idx])
			
			# mapping = compute_mapping(corpus_embs[:,important_idx], query_emb[important_idx], corpus_arg_idxs[:,important_idx], query_arg_idx[important_idx])
			
			# mapping = torch.where(corpus_arg_idxs[:,important_idx]==query_arg_idx[important_idx], corpus_embs[:,important_idx],torch.zeros_like(corpus_embs[:,important_idx]))
			# mapping = torch.sum(mapping*query_emb[important_idx], axis=1)

			mapping = (corpus_arg_idxs[:,important_idx]==query_arg_idx[:,important_idx])*(corpus_embs,torch.zeros_like(corpus_embs[:,important_idx]))
			mapping = torch.sum(mapping*query_emb, axis=1)


			candidates = torch.argsort(mapping, descending=True)[:10*topk]
			del mapping
			torch.cuda.empty_cache()

			candidate_embs = torch.where((corpus_arg_idxs[candidates,:]==query_arg_idx),corpus_embs[candidates,:],torch.zeros_like(corpus_embs[candidates,:]))
			scores = (candidate_embs*query_emb).sum(axis=1)
			sort_idx = torch.argsort(scores, descending=True)[:topk]
			sort_candidates = candidates[sort_idx]
			sort_scores = scores[sort_idx]
			partial_scores[qidx]=sort_scores.tolist()
			partial_results[qidx]=sort_candidates.tolist()
			# print(i)
			pbar.update(10 * i + 1)
			# if i==5:
			# 	break
		all_results.update(partial_results)
		all_scores.update(partial_scores)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--corpus_emb_path", type=str, required=True, help='The embedding file or the dir to save all the files with .tf')
	parser.add_argument("--query_emb_path", type=str, required=True, help='The embedding file or the dir to save all the files with .tf')
	parser.add_argument("--vocab_path", type=str, required=True, help='in case when using token embedding maxsim search instead of pooling embedding')
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--M", type=float, default=0.1)
	parser.add_argument("--passages_per_file", type=int, default=1000000, help='our default tf record include 1000,000 passages per file')
	parser.add_argument("--data_type", type=str, default='16', help='16 or 32 bit')
	parser.add_argument("--index", action='store_true')
	parser.add_argument("--add_cls", action='store_true')
	parser.add_argument("--max_passage_each_index", type=int, default=None, help='Set a passage number limitation for index')
	parser.add_argument("--id_to_doc_path", type=str, default=None)
	parser.add_argument("--index_path", type=str, required=True)
	args = parser.parse_args()

	vocab_dict = {}
	with open(args.vocab_path, 'r') as f:
		for i, line in enumerate(f):
			vocab = line.strip()
			vocab_dict[vocab]=i

	corpus_files=[]
	if os.path.isdir(args.corpus_emb_path):
		corpus_files = glob.glob(os.path.join(args.corpus_emb_path, '*.jsonl.gz'))
	else:
		corpus_files = [args.corpus_emb_path]


	
	if not os.path.exists(args.index_path):
		os.mkdir(args.index_path)
	print('Load %d tfrecord files...'%(len(corpus_files)))
	index_file = os.path.join(args.index_path,'index.pickle')
	load_jsonl_and_index(corpus_files, \
							data_num=args.passages_per_file, \
							dim=args.emb_dim, vocab_dict=vocab_dict, data_type=args.data_type, \
							add_cls=args.add_cls, index=args.index, save_path=index_file)
	
	

	print('finish')


if __name__ == "__main__":
	main()