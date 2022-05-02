import argparse
import os
import glob
# os.environ['OMP_NUM_THREADS'] = str(32)
import numpy as np
import math
from progressbar import *
# from util import load_tfrecords_and_index, read_id_dict, faiss_index
from multiprocessing import Pool, Manager
import pickle5 as pickle
import torch
import torch.nn as nn
torch.cuda.set_device(0)
import time
import faiss



def faiss_search(query_embs, corpus_embs, batch=1, topk=1000):
	print('start faiss index')
	query_embs = np.concatenate([query_embs,query_embs], axis=1)
	corpus_embs = np.concatenate([corpus_embs,corpus_embs], axis=1)

	dimension = query_embs.shape[1]
	res = faiss.StandardGpuResources()
	res.noTempMemory()
	# res.setTempMemory(1000 * 1024 * 1024) # 1G GPU memory for serving query
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.device = 0
	flat_config.useFloat16=True
	index = faiss.GpuIndexFlatIP(res, dimension, flat_config)

	print("Load index to GPU...")
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
	# parser.add_argument("--corpus_emb_path", type=str, required=True, help='The embedding file or the dir to save all the files with .tf')
	parser.add_argument("--query_emb_path", type=str, required=False, help='The embedding file or the dir to save all the files with .tf')
	parser.add_argument("--index_prefix", type=str, default='msmarco-passage')
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--M", type=float, default=0.1)
	parser.add_argument("--topk", type=int, default=1000)
	parser.add_argument("--combine_cls", action='store_true')
	parser.add_argument("--brute_force", action='store_true')
	parser.add_argument("--index_path", type=str, required=True)
	parser.add_argument("--faiss_index_path", type=str)
	parser.add_argument("--use_gpu", action='store_true')
	parser.add_argument("--rerank", action='store_true')
	parser.add_argument("--fusion", action='store_true')
	parser.add_argument("--lamda", type=float, default=1)
	parser.add_argument("--total_shrad", type=int, default=1)
	parser.add_argument("--shrad", type=int, default=0)
	parser.add_argument("--run_name", type=str, default='h2oloo')
	args = parser.parse_args()

	if not args.use_gpu:
		import mkl
		mkl.set_num_threads(36)


	
	query_texts = []
	query_ids = []
	qid2qidx = {}

	with open(args.query_emb_path, 'rb') as f:
		query_embs, query_arg_idxs, qids=pickle.load(f)



	if args.fusion:
		args.combine_cls = True
		cpu_index = faiss.read_index(os.path.join(args.faiss_index_path, 'index'))
		print('Reconstruct corpus vector from index...')
		vector_num = cpu_index.ntotal
		corpus_dense_reps = cpu_index.reconstruct_n(0,vector_num)
		corpus_dense_reps = corpus_dense_reps.astype(np.float16)

		with open(os.path.join(args.faiss_index_path, 'embedding.pkl'), 'rb') as f:
			query_dense_reps = pickle.load(f)
			qid2query_dense_reps = {}
			for row_num in range(query_dense_reps.shape[0]):
				qid = query_dense_reps.iloc[row_num,0]
				query_dense_rep = query_dense_reps.iloc[row_num,2].astype(np.float16)
				qid2query_dense_reps[qid] = query_dense_rep
			query_dense_reps = []
			for qid in qids:
				query_dense_reps.append(np.expand_dims(qid2query_dense_reps[qid], axis=0))
			query_dense_reps = np.concatenate(query_dense_reps, axis=0)

		query_embs = np.concatenate([query_embs, args.lamda*query_dense_reps], axis=1).astype(np.float16) #concat qid in to embeddings

	# else:
	# 	query_embs = np.concatenate([np.expand_dims(qids, axis=1), query_embs], axis=1).astype(np.float16) #concat qid in to embeddings

	if args.use_gpu:
		query_embs = torch.from_numpy(query_embs).cuda(0)
		query_arg_idxs = torch.from_numpy(query_arg_idxs).cuda(0)
	else:
		query_embs = torch.from_numpy(query_embs.astype(np.float32))
		query_arg_idxs = torch.from_numpy(query_arg_idxs)


	
	
	with open(args.index_path, 'rb') as f:
		corpus_embs, corpus_arg_idxs, docids=pickle.load(f)

		doc_num_per_shrad = len(docids)//args.total_shrad
		if args.shrad==(args.total_shrad-1):
			corpus_embs = corpus_embs[doc_num_per_shrad*args.shrad:]
			corpus_arg_idxs = corpus_arg_idxs[doc_num_per_shrad*args.shrad:]
			docids = docids[doc_num_per_shrad*args.shrad:]
			
			if args.fusion:
				docidxs = []
				for doc in docids:
					docidxs.append(int(doc))
				corpus_dense_reps = corpus_dense_reps[docidxs][doc_num_per_shrad*args.shrad:]
				corpus_embs = np.concatenate([corpus_embs, args.lamda*corpus_dense_reps], axis=1)
		else:
			corpus_embs = corpus_embs[doc_num_per_shrad*args.shrad:doc_num_per_shrad*(args.shrad+1)]
			corpus_arg_idxs = corpus_arg_idxs[doc_num_per_shrad*args.shrad:doc_num_per_shrad*(args.shrad+1)]
			docids = docids[doc_num_per_shrad*args.shrad:doc_num_per_shrad*(args.shrad+1)]
			if args.fusion:
				corpus_dense_reps = corpus_dense_reps[ocidxs][doc_num_per_shrad*args.shrad:doc_num_per_shrad*(args.shrad+1)]
				corpus_embs = np.concatenate([corpus_embs, args.lamda*corpus_dense_reps], axis=1)
		if args.use_gpu:
			corpus_embs = torch.from_numpy(corpus_embs).cuda(0)
			corpus_arg_idxs = torch.from_numpy(corpus_arg_idxs).cuda(0)
		else:
			corpus_embs = torch.from_numpy(corpus_embs.astype(np.float32)) 
			corpus_arg_idxs = torch.from_numpy(corpus_arg_idxs)
		# density = corpus_embs!=0
		# density = density.sum(axis=1)
		# print(torch.sum(density)/8841823/args.emb_dim)
		

	all_results = {}
	all_scores = {}

	# _,_ = faiss_search(query_embs[:,1:].astype(np.float32), corpus_embs.astype(np.float32))



	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*len(query_embs)).start()
	print('search ...')

	start_time = time.time()
	total_num_idx = 0
	for i, (query_emb, query_arg_idx) in enumerate(zip(query_embs, query_arg_idxs)):

		qidx = i
		if args.M==0:
			total_num_idx+=args.emb_dim
			candidate_sparse_embs = ((corpus_arg_idxs[:,:]==query_arg_idx)*corpus_embs[:,:args.emb_dim])					

			if args.combine_cls:
				candidate_dense_embs = corpus_embs[:,args.emb_dim:]
				scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[:args.emb_dim])) + torch.einsum('ij,j->i',(candidate_dense_embs, query_emb[args.emb_dim:]))
				del candidate_sparse_embs, candidate_dense_embs
			else:
				scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[:args.emb_dim]))
				del candidate_sparse_embs
			sort_idx = torch.argsort(scores, descending=True)[:args.topk]
			sort_candidates = sort_idx
			sort_scores = scores[sort_idx]

			
			torch.cuda.empty_cache()

		else:
			num_idx = int((query_emb[:args.emb_dim] > args.M).sum())
			if args.combine_cls:
				num_cls_idx = int((query_emb[args.emb_dim:] > args.M).sum())
				important_cls_idx = torch.argsort(query_emb[args.emb_dim:], axis=0, descending=True).tolist()[:num_cls_idx]
			if args.combine_cls:
				total_num_idx += num_idx + num_cls_idx
			else:
				total_num_idx += num_idx 
			if num_idx >40:
				num_idx=40
			if num_idx==0:
				num_idx=1
			important_idx = torch.argsort(query_emb[:args.emb_dim], axis=0, descending=True).tolist()[:num_idx]

			
			#Approximate GIN
			candidate_sparse_embs = ((corpus_arg_idxs[:,important_idx]==query_arg_idx[important_idx])*corpus_embs[:,important_idx])
			if args.combine_cls:

				candidate_dense_embs = corpus_embs[:,args.emb_dim:]
				partial_scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[important_idx])) + torch.einsum('ij,j->i',(candidate_dense_embs[:,important_cls_idx], query_emb[args.emb_dim:][important_cls_idx]))
				# partial_scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[important_idx])) + torch.einsum('ij,j->i',(candidate_dense_embs, query_emb[args.emb_dim:]))
			else:
				partial_scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[important_idx])) 

			# IN as an approximation ablation
			# if args.combine_cls:
			# 	candidate_sparse_embs = corpus_embs[:,:args.emb_dim]
			# 	candidate_dense_embs = corpus_embs[:,args.emb_dim:]

			# 	partial_scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[:args.emb_dim])) + 1*torch.einsum('ij,j->i',(candidate_dense_embs, query_emb[args.emb_dim:]))
			# else:
			# 	partial_scores = torch.einsum('ij,j->i',(corpus_embs, query_emb))

			if args.rerank:
				candidates = torch.argsort(partial_scores, descending=True)[:10*args.topk]
				candidate_sparse_embs = ((corpus_arg_idxs[candidates,:]==query_arg_idx)*corpus_embs[candidates,:args.emb_dim])
				# candidate_sparse_embs = torch.where((corpus_arg_idxs[candidates,:]==query_arg_idx),corpus_embs[candidates,:args.emb_dim],torch.zeros_like(corpus_embs[candidates,:args.emb_dim]))
				if args.combine_cls:
					candidate_dense_embs = corpus_embs[candidates,args.emb_dim:]
					scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[:args.emb_dim])) + torch.einsum('ij,j->i',(candidate_dense_embs, query_emb[args.emb_dim:]))
				else:
					scores = torch.einsum('ij,j->i',(candidate_sparse_embs, query_emb[:args.emb_dim]))

				sort_idx = torch.argsort(scores, descending=True)[:args.topk]
				sort_candidates = candidates[sort_idx]
				sort_scores = scores[sort_idx]

				del important_idx, candidates, candidate_sparse_embs, scores, sort_idx
				torch.cuda.empty_cache()
			else:
				sort_candidates = torch.argsort(partial_scores, descending=True)[:args.topk]
				sort_scores = partial_scores[sort_candidates]

		all_scores[qids[i]]=sort_scores.cpu().tolist()
		all_results[qids[i]]=sort_candidates.cpu().tolist()
		# print(i)
		pbar.update(10 * i + 1)
		# if i==5:
		# 	break
	average_num_idx = total_num_idx/query_embs.shape[0]
	time_per_query = (time.time() - start_time)/query_embs.shape[0]
	print('Retrieving {} queries ({:0.3f} s/query), average number of index use {}'.format(query_embs.shape[0], time_per_query, average_num_idx))

	# search(query_embs, query_arg_idxs, corpus_embs, corpus_arg_idxs, all_results, all_scores, args.topk)

	# search(query_embs, query_arg_idxs, corpus_embs, corpus_arg_idxs, args.topk)

	# num_workers = 10
	# pool = Pool(num_workers)
	# num_q_per_worker = query_embs.shape[0]//num_workers
	
	

	# for worker in range(num_workers):
	# 	if (worker+1)==num_workers:
	# 		pool.apply_async(search ,(query_embs[worker*num_q_per_worker:], query_arg_idxs, corpus_embs, corpus_arg_idxs, all_results, all_scores, args.topk))
	# 	else: 
	# 		pool.apply_async(search ,(query_embs[worker*num_q_per_worker:(worker+1)*num_q_per_worker], query_arg_idxs, corpus_embs, corpus_arg_idxs, all_results, all_scores, args.topk))
	# pool.close()
	# pool.join()


	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*len(query_embs)).start()

	print('write results ...')
	if args.total_shrad==1:
		fout = open('result.trec', 'w')
	else:
		fout = open('result{}.trec'.format(args.shrad), 'w')
	for i, query_id in enumerate(all_results):
		# query_id = query_ids[qidx]
		result = all_results[query_id]
		score = all_scores[query_id]
		for rank, docidx in enumerate(result):
			docid = docids[docidx]
			if (docid!=query_id):
				fout.write('{} Q0 {} {} {} {}\n'.format(query_id, docid, rank+1, score[rank], args.run_name))
		pbar.update(10 * i + 1)
	fout.close()


	
	

	print('finish')


if __name__ == "__main__":
	main()