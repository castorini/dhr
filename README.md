# A Dense Representation Framework for Lexical and Semantic Matching
The repo contains the code for our paper:
*[A Dense Representation Framework for Lexical and Semantic Matching](https://arxiv.org/pdf/2112.04666.pdf)* Sheng-Chieh Lin and Jimmy Lin. Our code is revised from [Tevatron](https://github.com/texttron/tevatron). 

# DHR and DLR Training and Inference on MSMARCO Passage ranking
## Data Preparation
We first preprocess the corpus, development queries and official training data in the json format. Each passage in the corpus is a line with the format: {"text_id": passage_id, "text": [vocab_ids]}. Similarly, each query in the development set is a line with the format: {"text_id": query_id, "text": [vocab_ids]}. As for training data, we rearrange the official training data in the format: {"query": [vocab_ids], "positive_pids": [positive_passage_id0, positive_passage_id1, ...], "negative_pids": [negative_passage_id0, negative_passage_id1, ...]}. Note that we use string type for passage and query. You can also download our preprocessed data on [huggingface hub](https://huggingface.co/datasets/jacklin/msmarco_passage_ranking/tree/main).

## Simple Training
This below script is the DHR (DLR) training in our paper. You can simply switch ${MODEL} from DHR to DLR, and the option `--combine_cls` would be turned off automatically.
```shell=bash
export CUDA_VISIBLE_DEVICES=0
export MODEL=DHR
export CLSDIM=128
export DLRDIM=768
export MODEL_DIR=${MODEL}_CLS${CLSDIM}
export DATA_DIR=need_your_assignment

python -m tevatron.driver.train \
  --output_dir ${MODEL_DIR} \
  --train_dir ${DATA_DIR}/official_train \
  --corpus_dir ${DATA_DIR}/corpus \
  --model_name_or_path distilbert-base-uncased  \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 24 \
  --learning_rate 7e-6 \
  --q_max_len 32 \
  --p_max_len 150 \
  --num_train_epochs 6 \
  --add_pooler \
  --model ${MODEL} \
  --projection_out_dim ${CLSDIM} \
  --train_n_passages 8 \
  --dataloader_num_workers 8 \
  --combine_cls \
```

## Inference MSMARCO Passage for Retrieval
```
export CUDA_VISIBLE_DEVICES=0
export MODEL=DHR
export CLSDIM=128
export DLRDIM=768
export MODEL_DIR=${MODEL}_CLS${CLSDIM}
export CORPUS=msmarco-passage
export SPLIT=dev.small
export INDEX_DIR=${MODEL_DIR}/encoding${DLRDIM}
export DATA_DIR=need_your_assignment

# Corpus
for i in $(seq -f "%02g" 0 10)
do
  echo '============= Inference doc.split '${i} ' ============='
  srun --gres=gpu:p100:1 --mem=16G --cpus-per-task=2 --time=1:40:00 \
  python -m tevatron.driver.encode \
    --output_dir ${MODEL_DIR} \
    --model_name_or_path ${MODEL_DIR} \
    --add_pooler \
    --projection_out_dim ${CLSDIM} \
    --dlr_out_dim ${DLRDIM} \
    --combine_cls \
    --model ${MODEL} \
    --fp16 \
    --p_max_len 150 \
    --per_device_eval_batch_size 128 \
    --encode_in_path ${DATA_DIR}/corpus/split${i}.json \
    --encoded_save_path ${INDEX_DIR}/${CORPUS}.split${i}.pt &
done

# Merge index
python -m retrieval.index \
  --index_path ${INDEX_DIR} \
  --index_prefix ${CORPUS}
mkdir ${INDEX_DIR}/index
mv ${INDEX_DIR}/${CORPUS}.index.pt ${INDEX_DIR}/index/

# Queries
for SPLIT in dev.small
do
  mkdir ${INDEX_DIR}/queries
  python -m tevatron.driver.encode \
    --output_dir ${MODEL_DIR} \
    --model_name_or_path ${MODEL_DIR} \
    --fp16 \
    --q_max_len 32 \
    --model ${MODEL} \
    --encode_is_qry \
    --combine_cls \
    --add_pooler \
    --projection_out_dim ${CLSDIM} \
    --dlr_out_dim ${DLRDIM} \
    --per_device_eval_batch_size 128 \
    --encode_in_path ${DATA_DIR}/queries/queries.${SPLIT}.json \
    --encoded_save_path ${INDEX_DIR}/queries/queries.${CORPUS}.${SPLIT}.pt
done
```
## End-to-end Retrieval with GIP
```
# GIP retrieval
for shrad in 0
do
  echo 'run shrad'$shrad
  python -m retrieval.gip_retrieval \
    --query_emb_path ${INDEX_DIR}/queries/queries.${CORPUS}.${SPLIT}.pt \
    --emb_dim ${DLRDIM} \
    --index_path ${INDEX_DIR}/index/${CORPUS}.index.pt \
    --topk 1000 \
    --total_shrad 1 \
    --shrad $shrad \
    --theta 0.3 \
    --rerank \
    --use_gpu \
    --combine_cls \
done
```
## Evaluation
The run file, result.trec, is in the trec format so that you can directly evaluate the result using pyserini.
```
python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank ${QREL_PATH} result.trec
python -m pyserini.eval.trec_eval -c -m recall.1000 ${QREL_PATH} result.trec
```
# BEIR Evaluation
We provide two scripts for BEIR evaluation and use trec-covid as an example.
## Evaluation with GIP
We first downlaod beir dataset; then, tokenize corpus and queries, and convert qrel in trec format. The files are in `dataset/trec-covid`.
```
export CORPUS=trec-covid
export SPLIT=test
python -m tevatron.beir.preprocess --dataset ${CORPUS}
```
Then we tokenize the query and corpus.
```
python -m tevatron.utils.tokenize_corpus \
  --corpus_path ./dataset/${CORPUS}/corpus/collection.json \
  --output_dir ./dataset/${CORPUS}/tokenized_data/corpus \
  --corpus_domain beir \
  --tokenize --encode

python -m tevatron.utils.tokenize_query \
  --qry_file ./dataset/${CORPUS}/queries/queries.${SPLIT}.tsv \
  --output_dir ./dataset/${CORPUS}/tokenized_data/queries

```
Following the above inference scripts for msmarco-passage data, we run inference, GIP retrieval and evaluation on the BEIR dataset.
```
export CUDA_VISIBLE_DEVICES=0
export MODEL=DHR
export CLSDIM=128
export DLRDIM=768
export MODEL_DIR=${MODEL}_CLS${CLSDIM}
export CORPUS=trec-covid
export SPLIT=test
export INDEX_DIR=${MODEL_DIR}/encoding${DLRDIM}
export DATA_DIR=./dataset/${CORPUS}/tokenized_data

# Corpus
for file in ${DATA_DIR}/corpus/split*.json
do
  i=$(echo $file |rev | cut -c -7 |rev | cut -c -2 )
  echo "===========inference ${file}==========="
  python -m tevatron.driver.encode \
    --output_dir ${MODEL_DIR} \
    --model_name_or_path ${MODEL_DIR} \
    --projection_out_dim ${CLSDIM} \
    --dlr_out_dim ${DLRDIM} \
    --model ${MODEL} \
    --add_pooler \
    --combine_cls \
    --fp16 \
    --p_max_len 512 \
    --per_device_eval_batch_size 32 \
    --encode_in_path ${file} \
    --encoded_save_path ${INDEX_DIR}/${CORPUS}.split${i}.pt
done

# Merge index
python -m retrieval.index \
  --index_path ${INDEX_DIR} \
  --index_prefix ${CORPUS}
mkdir ${INDEX_DIR}/index
mv ${INDEX_DIR}/${CORPUS}.index.pt ${INDEX_DIR}/index/

# QUERY
mkdir ${INDEX_DIR}/queries
python -m tevatron.driver.encode \
  --output_dir ${MODEL_DIR} \
  --model_name_or_path ${MODEL_DIR} \
  --fp16 \
  --q_max_len 512 \
  --model ${MODEL} \
  --encode_is_qry \
  --combine_cls \
  --add_pooler \
  --projection_out_dim ${CLSDIM} \
  --dlr_out_dim ${DLRDIM} \
  --per_device_eval_batch_size 128 \
  --encode_in_path ${DATA_DIR}/queries/queries.${SPLIT}.json \
  --encoded_save_path ${INDEX_DIR}/queries/queries.${CORPUS}.${SPLIT}.pt

```
```
# GIP retrieval
for shrad in 0
do
  echo 'run shrad'$shrad
  python -m retrieval.gip_retrieval \
    --query_emb_path ${INDEX_DIR}/queries/queries.${CORPUS}.${SPLIT}.pt \
    --emb_dim ${DLRDIM} \
    --index_path ${INDEX_DIR}/index/${CORPUS}.index.pt \
    --topk 1000 \
    --total_shrad 1 \
    --shrad $shrad \
    --theta 0.3 \
    --rerank \
    --use_gpu \
    --combine_cls
done
```
```
# Evaluation
python -m pyserini.eval.trec_eval -c -mndcg_cut.10 -mrecall.100 ./dataset/${CORPUS}/qrels/qrels.${SPLIT}.tsv result.trec
python -m retrieval.rcap_eval --qrel_file_path ./dataset/${CORPUS}/qrels/qrels.${SPLIT}.tsv --run_file_path result.trec

```
## Evaluation with Sentence Transformer
The second one is to directly use [BEIR](https://github.com/beir-cellar/beir) API to conduct brute-force search. No densification before retrieval; thus, the result can be considered the upper bound of our model. 
```
python -m tevatron.beir.encode_and_retrieval --dataset trec-covid --model ${MODEL_DIR}
```



