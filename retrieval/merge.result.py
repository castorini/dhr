import argparse
import pickle
import glob
import os
import numpy as np
from collections import defaultdict
from progressbar import *


	


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--total_shrad", type=int, default=1)
	parser.add_argument("--topk", type=int, default=1000)

	args = parser.parse_args()
	results = defaultdict(list)
	scores = defaultdict(list)
	for shrad in range(args.total_shrad):
		with open('result{:02d}.trec'.format(shrad), 'r') as f:
			for line in f:
				query_id, _, docid, rank, score, _ = line.strip().split(' ')
				score = float(score)
				results[query_id].append(docid)
				scores[query_id].append(score)
	

	print('write results ...')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
			' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*len(results)).start()
	fout = open('result.trec', 'w')
	for i, query_id in enumerate(results):
		score = scores[query_id]
		result = results[query_id]
		sort_idx = np.array(score).argsort()[::-1][:args.topk]
		for rank, idx in enumerate(sort_idx):
			fout.write('{} Q0 {} {} {} {}\n'.format(query_id, result[idx], rank+1, score[idx], 'DWM'))
		pbar.update(10 * i + 1)
	fout.close()



if __name__ == "__main__":
	main()