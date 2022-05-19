import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
import argparse
from tqdm import tqdm
import os
import json
from multiprocessing import Pool
from transformers import AutoTokenizer
from .data_reader import create_dir

DATA_ITEM = {'msmarco-passage': {'id':'id', 'contents': ['contents']}, 
			 'beir': {'id':'_id', 'contents': ['title', 'text']}}

def tokenize_and_json_save(data_item, data_type, tokenizer, lines, jsonl_path, tokenize, encode):
    output = open(jsonl_path, 'w')
    for i, line in enumerate( tqdm(lines, total=len(lines), desc=f"write {output}") ):
        if data_type == 'tsv':
            docid, contents = line.strip().split('\t')
        elif (data_type =='json') or (data_type =='jsonl'):
            line = json.loads(line.strip())
            docid = line[data_item['id']]

            contents = []
            for content in data_item['contents']:
                contents.append(line[content])
            contents = ' '.join(contents)
        if tokenize:
            if encode:
                contents = tokenizer.encode(contents, add_special_tokens=False)
                # Fit the format of tevatron
                output_dict = {'text_id': docid, 'text': contents} 
            else:
                contents = ' '.join(tokenizer.tokenize(contents))
                output_dict = {'id': docid, 'contents': contents}
        else:
            output_dict = {'id': docid, 'contents': contents}
        output.write(json.dumps(output_dict) + '\n')
    output.close()

def main():
    parser = argparse.ArgumentParser(
        description='Transform corpus into wordpiece corpus')
    parser.add_argument('--corpus_path', required=True, help='TSV or json corpus file with format {docid}\t{document}.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--corpus_domain', required=False, default='msmarco-passage')
    parser.add_argument('--tokenizer', required=False, default='bert-base-uncased', help='tokenizer model name')
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--encode', action='store_true')
    parser.add_argument('--num_workers', type=int, required=False, default=None)
    parser.add_argument('--max_line_per_file', type=int, required=False, default=300000, help='max length 150 use default; max length 512 use 300000')
    args = parser.parse_args()

    if args.encode:
        if not args.tokenize:
            raise ValueError('if you want to encode, you must set tokenize option!')

    create_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    data_type = (args.corpus_path).split('.')[-1]
    if (data_type != 'tsv') and (data_type != 'json') and (data_type != 'jsonl'):
        raise ValueError('--corpus_path should be tsv, json or jsonl format')

    with open(args.corpus_path, 'r') as f:
        print("read {}".format(args.corpus_path))
        lines = f.readlines()
        total_num_docs = len(lines)
        print("total {} lines".format(total_num_docs))

    ## for debug
    # tokenize_and_json_save(DATA_ITEM[args.corpus_domain], data_type, tokenizer, lines, os.path.join(jsonl_dir, 'split.json'), args.tokenize )
    if args.num_workers is None:
        num_docs_per_worker = args.max_line_per_file
        args.num_workers = total_num_docs // num_docs_per_worker
        if (total_num_docs%num_docs_per_worker ) != 0:
            args.num_workers+=1
    else:
        num_docs_per_worker = total_num_docs//args.num_workers
        if (total_num_docs%args.num_workers) != 0:
            args.num_workers+=1

    logging.info(f'Run with {args.num_workers} workers on {total_num_docs} documents')        
    pool = Pool(args.num_workers)
    for i in range(args.num_workers):
        f_out = os.path.join(args.output_dir, 'split%02d.json'%i)
        start = i*num_docs_per_worker
        if i==(args.num_workers-1):
            pool.apply_async(tokenize_and_json_save ,(DATA_ITEM[args.corpus_domain], data_type, tokenizer,\
                             lines[start:], f_out, args.tokenize, args.encode))
        else:
            pool.apply_async(tokenize_and_json_save ,(DATA_ITEM[args.corpus_domain], data_type, tokenizer,\
                             lines[start:(start+num_docs_per_worker)], f_out, args.tokenize, args.encode))

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
