from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
import json
from tqdm import tqdm
import argparse
import itertools
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Extract text contents from anserini index')
	parser.add_argument('--index_path', required=True, help='anserini index path')
	parser.add_argument('--output_path', required=True, help='Output file in the anserini jsonl format.')
	parser.add_argument('--tf_only' , action='store_true')
	args = parser.parse_args()

	index_reader = IndexReader(args.index_path)
	searcher = SimpleSearcher(args.index_path)
	total_num_docs = searcher.num_docs
	
	term_dict = {}
	for idx, term in tqdm(enumerate(index_reader.terms()), desc=f"read index terms"): 
		term_dict[term.term] = idx
	
	fout = open(args.output_path, 'w')
	for i in tqdm(range(total_num_docs), total=total_num_docs, desc=f"compute bm25 vector"): 
		docid = searcher.doc(i).docid()
		tf = index_reader.get_document_vector(docid)
		vocab_ids = []
		weights = []
		for term, weight in tf.items():
			vocab_ids.append(term_dict[term])
			if args.tf_only:
				weights.append(weight)
			else:		
				weights.append(index_reader.compute_bm25_term_weight(docid, term, analyzer=None))

		output_dict = {'id': docid, 'token_id': vocab_ids, 'weights': weights}
		fout.write(json.dumps(output_dict) + '\n')
	fout.close()