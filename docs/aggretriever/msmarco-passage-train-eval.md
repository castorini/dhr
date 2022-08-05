# Training and Inference on MSMARCO Passage ranking
## Data Preparation
We first preprocess the corpus, development queries and official training data in the json format. Each passage in the corpus is a line with the format: `{"text_id": passage_id, "text": [vocab_ids]}`. Similarly, each query in the development set is a line with the format: `{"text_id": query_id, "text": [vocab_ids]}`. As for training data, we rearrange the official training data in the format: `{"query": [vocab_ids], "positive_pids": [positive_passage_id0, positive_passage_id1, ...], "negative_pids": [negative_passage_id0, negative_passage_id1, ...]}`. Note that we use string type for passage and query. You can also download our preprocessed data on huggingface hub: [official_train](https://huggingface.co/datasets/jacklin/msmarco_passage_ranking_corpus), [queries](https://huggingface.co/datasets/jacklin/msmarco_passage_ranking_queries) and [corpus](https://huggingface.co/datasets/jacklin/msmarco_passage_ranking_corpus).

## Simple Training
This below script is the Aggretriever training in our paper. Here we use distilbert-base-uncased as an example. You can switch to any backbone using `--model_name_or_path`.
```shell=bash
export CUDA_VISIBLE_DEVICES=0
export MODEL=AGG
export CLSDIM=128
export AGGDIM=640
export MODEL_DIR=${MODEL}_CLS${CLSDIM}XAGG${AGGDIM}
export DATA_DIR=need_your_assignment

python -m tevatron.driver.train \
  --output_dir ${MODEL_DIR} \
  --train_dir ${DATA_DIR}/official_train \
  --corpus_dir ${DATA_DIR}/corpus \
  --model_name_or_path distilbert-base-uncased  \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-6 \
  --q_max_len 32 \
  --p_max_len 128 \
  --num_train_epochs 3 \
  --add_pooler \
  --model ${MODEL} \
  --projection_out_dim ${CLSDIM} \
  --agg_dim ${AGGDIM}
  --train_n_passages 8 \
  --dataloader_num_workers 8 \
```

## Inference MSMARCO Passage for Retrieval
```
export CUDA_VISIBLE_DEVICES=0
export CORPUS=msmarco-passage
export SPLIT=dev.small
export INDEX_DIR=${MODEL_DIR}/encoding
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
    --agg_dim ${AGGDIM} \
    --model ${MODEL} \
    --fp16 \
    --p_max_len 128 \
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
    --add_pooler \
    --projection_out_dim ${CLSDIM} \
    --agg_dim ${AGGDIM} \
    --per_device_eval_batch_size 128 \
    --encode_in_path ${DATA_DIR}/queries/queries.${SPLIT}.json \
    --encoded_save_path ${INDEX_DIR}/queries/queries.${CORPUS}.${SPLIT}.pt
done
```
## End-to-end Retrieval
```
# IP retrieval
for shrad in 0
do
  echo 'run shrad'$shrad
  python -m retrieval.gip_retrieval \
    --query_emb_path ${INDEX_DIR}/queries/queries.${CORPUS}.${SPLIT}.pt \
    --index_path ${INDEX_DIR}/index/${CORPUS}.index.pt \
    --topk 1000 \
    --total_shrad 1 \
    --shrad $shrad \
    --IP \
    --use_gpu \
done
```
## Evaluation
The run file, result.trec, is in the trec format so that you can directly evaluate the result using pyserini.
```
python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank ${QREL_PATH} result.trec
python -m pyserini.eval.trec_eval -c -m recall.1000 ${QREL_PATH} result.trec
```


