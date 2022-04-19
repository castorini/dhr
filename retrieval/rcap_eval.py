import argparse
from .evaluation.custom_metrics import recall_cap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel_file_path", type=str, required=True)
    parser.add_argument("--run_file_path", type=str, required=True)
    parser.add_argument("--cutoff", type=int, default=100, required=False)
    args = parser.parse_args()

    qrels = {}
    with open(args.qrel_file_path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            if qid not in qrels:
                qrels[qid] = {docid: int(rel)}
            else:
                qrels[qid][docid] = int(rel)

    results = {}
    with open(args.run_file_path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split(' ')
            if qid not in results:
                results[qid] = {docid: float(score)}
            else:
                results[qid][docid] = float(score)

    print(recall_cap(qrels, results, [args.cutoff]))

if __name__ == "__main__":
    main()