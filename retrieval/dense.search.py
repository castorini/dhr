import argparse
import os
import faiss
import numpy as np
from util import load_tfrecords_and_index, load_term_weight_tfrecords
import pickle
import mkl
import time
from progressbar import *


def save_pickle(Distance, Index, filename):
	print('save pickle...')
	with open(filename, 'wb') as f:
		pickle.dump([Distance, Index], f)



class index_method():
	def __init__(self, dimension, GPU, gpu_device):
		self.GPU = GPU
		if GPU:
			res = faiss.StandardGpuResources()
			res.noTempMemory()
			# res.setTempMemory(1000 * 1024 * 1024) # 1G GPU memory for serving query
			flat_config = faiss.GpuIndexFlatConfig()
			flat_config.device = gpu_device
			flat_config.useFloat16=True
			self.index = faiss.GpuIndexFlatIP(res, dimension, flat_config)
		else:
			self.index = faiss.IndexFlatIP(dimension)
	def load_index(self, index_file):
		print("Read index:" +index_file+"...")
		cpu_index = faiss.read_index(index_file)
		if self.GPU:
			print('Reconstruct vector from index...')
			# faiss.downcast_index(cpu_index).make_direct_map()
			vector_num = cpu_index.ntotal
			vectors = cpu_index.reconstruct_n(0,vector_num)
			print("Load index ("+index_file+ ") to GPU...")
			self.index.add(vectors)
			del vectors
			del cpu_index
		else:
			self.index = cpu_index



def search(index, query_embs, batch_size, threads, topk, dimension):


	Distance = []
	Index = []
	batch = batch_size


	print("Search with batch size %d"%(batch))
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	           ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=query_embs.shape[0]//batch).start()
	start_time = time.time()

	for i in range(query_embs.shape[0]//batch):
		faiss.omp_set_num_threads(threads)

		D,I=index.search(query_embs[i*batch:(i+1)*batch], topk)


		Distance.append(D)
		Index.append(I)
		pbar.update(i + 1)

	faiss.omp_set_num_threads(threads)
	D,I=index.search(query_embs[(i+1)*batch:], topk)


	Distance.append(D)
	Index.append(I)

	time_per_query = (time.time() - start_time)/query_embs.shape[0]
	print('Retrieving {} queries ({:0.3f} s/query)'.format(query_embs.shape[0], time_per_query))
	Distance = np.concatenate(Distance, axis=0)
	Index = np.concatenate(Index, axis=0)
	return Distance, Index


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--query_word_num", type=int, default=1)
	parser.add_argument("--doc_word_num", type=int, default=1)
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--topk", type=int, default=1000)
	parser.add_argument("--batch_size", type=int, default=1, help='in order to measure time/query, we have to set batch size to query_word_num')
	parser.add_argument("--threads", type=int, default=1)
	parser.add_argument("--intermediate_path", type=str, required=True)
	parser.add_argument("--index_file", type=str, required=True)
	parser.add_argument("--query_emb_path", type=str)
	parser.add_argument("--data_type", type=str, default='16')
	parser.add_argument("--use_gpu", action='store_true')
	parser.add_argument("--gpu_device", type=int, default=0)
	args = parser.parse_args()
	if not os.path.exists(args.intermediate_path):
		os.mkdir(args.intermediate_path)
	query_embs, qids=load_tfrecords_and_index([args.query_emb_path],\
									data_num=800000, word_num=args.query_word_num, \
									dim=args.emb_dim, data_type=args.data_type)


	mkl.set_num_threads(args.threads)
	# query_embs0, qids0=load_term_weight_tfrecords([args.query_emb_path],\
	# 								dim=args.emb_dim, data_type=args.data_type)

	# query_embs=query_embs.reshape((-1, args.query_word_num, args.emb_dim))

	index = index_method(args.emb_dim, args.use_gpu, args.gpu_device)
	index.load_index(args.index_file)


	Distance, Index = search(index.index, query_embs, args.batch_size, args.threads, args.topk, args.emb_dim)

	Index=Index//args.doc_word_num
	Index = Index.reshape((-1, args.query_word_num*args.topk))
	Distance = Distance.reshape((-1, args.query_word_num*args.topk))
	pickle_file = 'shrad-'+args.index_file.split('-')[-1]
	try:
		save_pickle(Distance, Index, os.path.join(args.intermediate_path ,pickle_file))
	except:
		pickle_file = 'shrad-0'
		save_pickle(Distance, Index, os.path.join(args.intermediate_path ,pickle_file))
	print('finish')

if __name__ == "__main__":
	main()