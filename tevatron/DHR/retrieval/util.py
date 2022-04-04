import os
import pickle

# import mkl
# mkl.set_num_threads(16)
import numpy as np
import tensorflow.compat.v1 as tf
from numpy import linalg as LA
from progressbar import *
from collections import defaultdict
import glob
from scipy.sparse import csc_matrix
import gzip
import json

def read_pickle(filename):
	with open(filename, 'rb') as f:
		Distance, Index=pickle.load(f)
	return Distance, Index


def read_id_dict(path):
	if os.path.isdir(path):
		files = glob.glob(os.path.join(path, '*.id'))
	else:
		files = [path]

	idx_to_id = {}
	id_to_idx = {}
	for file in files:
		f = open(file, 'r')
		for i, line in enumerate(f):
			try:
				idx, Id =line.strip().split('\t')
				idx_to_id[int(idx)] = Id
				id_to_idx[Id] = int(idx)
			except:
				Id = line.strip()
				idx_to_id[i] = Id
				# if len(Id.split(' '))==1:

				# else:
				# 	print(line+' has no id')
	return idx_to_id, id_to_idx

def write_result(qidxs, Index, Score, file, idx_to_qid, idx_to_docid, topk=None, run_name='Faiss'):
	print('write results...')
	with open(file, 'w') as fout:
		for i, qidx in enumerate(qidxs):
			try:
				qid = idx_to_qid[qidx]
			except:
				qid = qidx
			if topk==None:
				docidxs=Index[i]
				scores=Score[i]
				for rank, docidx in enumerate(docidxs):
					try:
						docid = idx_to_docid[docidx]
					except:
						docid = docidx
					fout.write('{} Q0 {} {} {} {}\n'.format(qid, docid, rank + 1, scores[rank], run_name))
			else:
				try:
					hit=min(topk, len(Index[i]))
				except:
					print('debug')

				docidxs=Index[i]
				scores=Score[i]
				for rank, docidx in enumerate(docidxs[:hit]):
					try:
						docid = idx_to_docid[docidx]
					except:
						docid = docidx
					fout.write('{} Q0 {} {} {} {}\n'.format(qid, docid, rank + 1, scores[rank], run_name))


def faiss_index(corpus_embs, docids, save_path, index_method):

	dimension=corpus_embs.shape[1]
	print("Indexing ...")
	if index_method==None or index_method=='flatip':
		cpu_index = faiss.IndexFlatIP(dimension)
		
	elif index_method=='hsw':
		cpu_index = faiss.IndexHNSWFlat(dimension, 256, faiss.METRIC_INNER_PRODUCT)
		cpu_index.hnsw.efConstruction = 256
	elif index_method=='quantize': # still try better way for balanced efficiency and effectiveness
		cpu_index = faiss.IndexHNSWPQ(dimension, 192, 256)
		cpu_index.hnsw.efConstruction = 256
		cpu_index.metric_type = faiss.METRIC_INNER_PRODUCT
		# ncentroids = 1000
		# code_size = dimension//4
		# cpu_index = faiss.IndexIVFPQ(cpu_index, dimension, ncentroids, code_size, 8)
		# cpu_index = faiss.IndexPQ(dimension, code_size, 8)
		# cpu_index = faiss.index_factory(768, "OPQ128,IVF4096,PQ128", faiss.METRIC_INNER_PRODUCT)
		# cpu_index = faiss.IndexIDMap(cpu_index)
		# cpu_index = faiss.GpuIndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_16bit_direct, faiss.METRIC_INNER_PRODUCT)
		

	cpu_index.verbose = True
	cpu_index.add(corpus_embs)
	if index_method=='quantize':
		print("Train index...")
		cpu_index.train(corpus_embs)
	print("Save Index {}...".format(save_path))
	faiss.write_index(cpu_index, save_path)

def save_pickle(corpus_embs, arg_idxs, docids, filename):
	print('save pickle...')
	with open(filename, 'wb') as f:
		pickle.dump([corpus_embs, arg_idxs, docids], f, protocol=4)

def load_tfrecords_and_index(srcfiles, data_num, word_num, dim, data_type, add_cls, index=False, save_path=None, batch=10000):
	def _parse_function(example_proto):
		features = {'doc_emb': tf.FixedLenFeature([],tf.string) , #tf.FixedLenSequenceFeature([],tf.string, allow_missing=True),
					'argx_id_id': tf.FixedLenFeature([],tf.string) ,
					'docid': tf.FixedLenFeature([],tf.int64)}
		parsed_features = tf.parse_single_example(example_proto, features)
		arg_idx = tf.decode_raw(parsed_features['argx_id_id'], tf.uint8)
		if data_type=='16':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float16)
		elif data_type=='32':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float32)
		docid = tf.cast(parsed_features['docid'], tf.int32)
		return corpus, arg_idx, docid
	print('Read embeddings...')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*data_num*len(srcfiles)).start()
	with tf.Session() as sess:
		docids=[]
		if add_cls:
			segment=2
		else:
			segment=1
		#assign memory in advance so that we can save memory without concatenate
		arg_idxs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.uint8)
		if (data_type=='16'): # Faiss now only support index array with float32
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim*segment), dtype=np.float16)
		elif data_type=='32':
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim*segment), dtype=np.float32)
		# else:
		# 	raise Exception('Please assign datatype 16 or 32 bits')
		counter = 0
		i = 0

		for srcfile in srcfiles:
			try:
				dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
			except:
				print('Cannot find data')
				continue
			dataset = dataset.map(_parse_function) # parse data into tensor
			dataset = dataset.repeat(1)
			dataset = dataset.batch(batch)
			iterator = dataset.make_one_shot_iterator()
			next_data = iterator.get_next()

			while True:
				try:
					corpus_emb, arg_idx, docid = sess.run(next_data)

					corpus_emb = corpus_emb.reshape(-1, dim*segment)

					sent_num = corpus_emb.shape[0]
					corpus_embs[counter:(counter+sent_num)] = corpus_emb
					arg_idxs[counter:(counter+sent_num)] = arg_idx

					docids+=docid.tolist()
					counter+=sent_num
					pbar.update(10 * i + 1)
					i+=sent_num
				except tf.errors.OutOfRangeError:
					break

		docids = np.array(docids).reshape(-1)
		corpus_embs = (corpus_embs[:len(docids)])
		arg_idxs = (arg_idxs[:len(docids)])
		mask = docids!=-1
		docids = docids[mask]
		corpus_embs = corpus_embs[mask]
		arg_idxs = arg_idxs[mask]
	if index:
		save_pickle(corpus_embs, arg_idxs, docids, save_path)
	else:
		return corpus_embs, arg_idxs, docids

def load_jsonl_and_index(srcfiles, data_num, dim, vocab_dict, data_type, add_cls, index=False, save_path=None, batch=10000):
	print('Count line...')
	data_num = 0
	for srcfile in srcfiles:
		with gzip.open(srcfile, 'rb') as f:
			for l in f:
				data_num+=1
	print('Total {} lines'.format(data_num))
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*data_num*len(srcfiles)).start()
	docids=[]
	if add_cls:
		segment=2
	else:
		segment=1
	#assign memory in advance so that we can save memory without concatenate
	arg_idxs = np.zeros((data_num , dim), dtype=np.uint8)
	if (data_type=='16'): # Faiss now only support index array with float32
		corpus_embs = np.zeros((data_num , dim*segment), dtype=np.float16)
	elif data_type=='32':
		corpus_embs = np.zeros((data_num , dim*segment), dtype=np.float32)
	# else:
	# 	raise Exception('Please assign datatype 16 or 32 bits')
	counter = 0
	i = 0


	for srcfile in srcfiles:
		
		with gzip.open(srcfile, "rb") as f:
			for line in f:
				data = json.loads(line.strip())
				embedding =np.zeros((30522), dtype=np.float16)
				for vocab, term_weight in data['vector'].items():
					embedding[vocab_dict[vocab]] = term_weight/100

				embedding = np.reshape(embedding[570:],(-1, dim))
				corpus_emb = embedding.max(0)
				arg_idx = embedding.argmax(0)
				docid = int(data['id'])


				corpus_emb = corpus_emb.reshape(-1, dim*segment)

				sent_num = corpus_emb.shape[0]
				corpus_embs[counter:(counter+sent_num)] = corpus_emb
				arg_idxs[counter:(counter+sent_num)] = arg_idx

				docids+=[docid]
				counter+=sent_num
				pbar.update(10 * i + 1)
				i+=sent_num


	docids = np.array(docids).reshape(-1)
	corpus_embs = (corpus_embs[:len(docids)])
	arg_idxs = (arg_idxs[:len(docids)])
	mask = docids!=-1
	docids = docids[mask]
	corpus_embs = corpus_embs[mask]
	arg_idxs = arg_idxs[mask]
	if index:
		save_pickle(corpus_embs, arg_idxs, docids, save_path)
	else:
		return corpus_embs, arg_idxs, docids

def load_tfrecords_and_analyze(srcfiles, data_num, word_num, dim, data_type, batch=1):
	def _parse_function(example_proto):
		features = {#'doc_emb': tf.FixedLenFeature([],tf.string) , #tf.FixedLenSequenceFeature([],tf.string, allow_missing=True),
					'id_p1': tf.FixedLenSequenceFeature([],tf.int64, allow_missing=True) ,
					'docid': tf.FixedLenFeature([],tf.int64)}
		parsed_features = tf.parse_single_example(example_proto, features)
		vocab_ids = tf.cast(parsed_features['id_p1'], tf.int32)
		docid = tf.cast(parsed_features['docid'], tf.int32)
		return vocab_ids, docid
	print('Read embeddings...')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*data_num*len(srcfiles)).start()
	with tf.Session() as sess:
		docids=[]
		segment=1
		# else:
		# 	raise Exception('Please assign datatype 16 or 32 bits')
		counter = 0
		i = 0
		vocab_adj =np.zeros((30522,30522), dtype=np.uint32)
		vocab_freq = np.zeros((30522), dtype=np.uint32)
		for srcfile in srcfiles:
			try:
				dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
			except:
				print('Cannot find data')
				continue
			dataset = dataset.map(_parse_function) # parse data into tensor
			dataset = dataset.repeat(1)
			dataset = dataset.batch(batch)
			iterator = dataset.make_one_shot_iterator()
			next_data = iterator.get_next()
			
			while True:
				try:
					vocab_ids, docid = sess.run(next_data)

					vocab_id_list = vocab_ids.squeeze().tolist()
					try:
						num_vocab_id = len(vocab_id_list)
						if num_vocab_id >1:
							for m in range(num_vocab_id):
								vocab_freq[vocab_id_list[m]]+=1
								for n in range(m+1,num_vocab_id,1):
									vocab_adj[vocab_id_list[m], vocab_id_list[n]]+=1

					except:
						vocab_freq[vocab_id_list]+=1



					pbar.update(10 * i + 1)
					i+=1
					# if i>=20000:
					# 	break
				except tf.errors.OutOfRangeError:
					break


		return vocab_freq, vocab_adj