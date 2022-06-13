import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
import argparse
import pathlib, os
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
logger = logging.getLogger(__name__)
from ...utils.data_reader import create_dir


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output_dir", required=False, default='./dataset', type=str)
	parser.add_argument("--dataset", required=True, type=str, help="beir dataset name")
	parser.add_argument("--split", default='test', type=str, help="beir dataset name")
	args = parser.parse_args()

	#### Download scifact.zip dataset and unzip the dataset
	create_dir(os.path.join('./download'))
	create_dir(os.path.join(args.output_dir))
	dataset = args.dataset
	url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
	data_path = util.download_and_unzip(url, './download')

	#### Provide the data_path where scifact has been downloaded and unzipped
	corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)

	create_dir(os.path.join(args.output_dir, args.dataset, 'corpus'))
	os.rename(os.path.join('./download', args.dataset, 'corpus.jsonl'), os.path.join(args.output_dir, args.dataset, 'corpus', 'collection.json'))

	create_dir(os.path.join(args.output_dir, args.dataset,'qrels'))
	qrel_fout = open(os.path.join(args.output_dir, args.dataset,'qrels', 'qrels.' + args.split + '.tsv'), 'w')

	create_dir(os.path.join(args.output_dir, args.dataset,'queries'))
	query_fout = open(os.path.join(args.output_dir, args.dataset, 'queries', 'queries.' + args.split + '.tsv'), 'w')

	for qid, answer in qrels.items():
		for docid, rel in answer.items():
			qrel_fout.write('{}\tQ0\t{}\t{}\n'.format(qid, docid, rel))
		query_fout.write('{}\t{}\n'.format(qid, queries[qid]))

	qrel_fout.close()
	query_fout.close()

if __name__ == "__main__":
	main()