################################################################################################################
# The evaluation code is revised from SPLADE repo: https://github.com/naver/splade/blob/main/src/beir_eval.py


import argparse
from .sentence_bert import DHR, SentenceTransformerModel
from transformers import AutoModelForMaskedLM, AutoTokenizer





from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()


    model_type_or_dir = args.model

    # loading model and tokenizer
    model = DHR(model_type_or_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir, use_fast=False)
    sentence_transformer = SentenceTransformerModel(model, tokenizer, args.max_length)


    dataset = args.dataset

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "dataset/{}".format(dataset)
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
    # data folder would contain these files: 
    # (1) nfcorpus/corpus.jsonl  (format: jsonlines)
    # (2) nfcorpus/queries.jsonl (format: jsonlines)
    # (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.retrieval.evaluation import EvaluateRetrieval

    dres = DRES(sentence_transformer)
    retriever = EvaluateRetrieval(dres, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 10, 100, 1000], metric="r_cap")
    res = {"NDCG@10": ndcg["NDCG@10"],
           "Recall@100": recall["Recall@100"],
           "R_cap@100": results2["R_cap@100"]}
    print("res for {}:".format(dataset), res, flush=True)


if __name__ == "__main__":
    main()