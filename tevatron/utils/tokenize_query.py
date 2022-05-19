import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
import argparse
from tqdm import tqdm
import os
import json
from collections import defaultdict
from transformers import AutoTokenizer
import sys
from .data_reader import read_tsv, create_dir

def main():
    parser = argparse.ArgumentParser(
        description='Tokenize query')
    parser.add_argument('--qry_file', required=True, help='format {qid}\t{qry}')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tokenizer', required=False, default='bert-base-uncased', help='tokenizer model name')
    args = parser.parse_args()

    create_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    qid2qry = read_tsv(args.qry_file)

    query_name = args.qry_file.split('/')[-1].replace('.tsv','.json')
    output_path = os.path.join(args.output_dir, query_name)
    output = open(output_path, 'w')
    with open(args.qry_file, 'r') as f:
        for line in tqdm(f, desc=f"tokenize query: {output_path}"):
            qid, qry = line.strip().split('\t')
            qry = tokenizer.encode(qry, add_special_tokens=False)
            output_dict = {"text_id": qid, "text": qry}
            output.write(json.dumps(output_dict) + '\n')
    output.close()
if __name__ == "__main__":
    main()