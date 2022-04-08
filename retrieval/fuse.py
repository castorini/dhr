import argparse
from collections import defaultdict
import time
from progressbar import *

parser = argparse.ArgumentParser(description='Rank list fusion')
parser.add_argument('--rank_file0', default=None, help='rank file for dense retrieval with format: qid\tdocid\trank\tscore')
parser.add_argument('--rank_file1', default=None, help='rank file for sparse retrieval with format: qid\tdocid\trank\tscore')
parser.add_argument('--topk', default=1000, type=int, help='number of hits to retrieve')
parser.add_argument('--output', required=True)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--run_name", type=str, default='fusion')

args = parser.parse_args()



def read_rank_list(file):
    qid_docid_list = defaultdict(list)
    qid_score_list = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _=line.strip().split(' ') 
            qid_docid_list[qid].append(docid)
            qid_score_list[qid].append(float(score))
    return qid_docid_list, qid_score_list

def fuse(docid_list0, docid_list1, doc_score_list0, doc_score_list1, alpha):
    score = defaultdict(float)
    score0 = defaultdict(float)
    for i, docid in enumerate(docid_list0):
        score0[docid]+=doc_score_list0[i]
    min_val0 = min(doc_score_list0)
    min_val1 = min(doc_score_list1)
    for i, docid in enumerate(docid_list1):
        if score0[docid]==0:
            score[docid]+=min_val0 + doc_score_list1[i]*alpha
        else:
            score[docid]+=doc_score_list1[i]*alpha
    for i, docid in enumerate(docid_list0):
        if score[docid]==0:
            score[docid]+=min_val1*alpha
        score[docid]+=doc_score_list0[i]
    score= {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
    return score

def main():

    print('Read ranked list0...')
    qid_docid_list0, qid_score_list0 = read_rank_list(args.rank_file0)
    print('Read ranked list1...')
    qid_docid_list1, qid_score_list1 = read_rank_list(args.rank_file1)

    qids = qid_docid_list0.keys()
    fout = open(args.output, 'w')
    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=len(qids)).start()
    start_time = time.time()
    for j, qid in enumerate(qids):
        rank_doc_score = fuse(qid_docid_list0[qid], qid_docid_list1[qid], qid_score_list0[qid], qid_score_list1[qid], args.alpha)
        for rank, doc in enumerate(rank_doc_score):
            if rank==args.topk:
                break
            score = rank_doc_score[doc]
            fout.write('{} Q0 {} {} {} {}\n'.format(qid, doc, rank + 1, score, args.run_name))
        pbar.update(j + 1)
    time_per_query = (time.time() - start_time)/len(qids)
    print('Fusing {} queries ({:0.3f} s/query)'.format(len(qids), time_per_query))
    fout.close()

    print('finish')

if __name__ == "__main__":
    main()
