# Training and Inference on MSMARCO Passage ranking
In the following, we describe how to train, encode and retrieve with DHR on MS MARCO passage-v1.
1. [MS MARCO Passage-v1 Data Preparation](#msmarco_data_prep)
1. [Training](#training)
1. [Generate Passage and Query Embeddings](#generate_embeddings)
1. [End-To-End Retrieval](#retrieval)
    1. [Retrieval on GPU](#retrieval_on_gpu)
    1. [Retrieval on CPU](#retrieval_on_cpu)
1. [Evaluation](#evaluation)


## MS MARCO Passage-v1 Data Preparation <a name="msmarco_data_prep"></a>
We first preprocess the corpus, development queries and official training data in the json format. Each passage in the corpus is a line with the format: `{"text_id": passage_id, "text": [vocab_ids]}`. Similarly, each query in the development set is a line with the format: `{"text_id": query_id, "text": [vocab_ids]}`. As for training data, we rearrange the official training data in the format: `{"query": [vocab_ids], "positive_pids": [positive_passage_id0, positive_passage_id1, ...], "negative_pids": [negative_passage_id0, negative_passage_id1, ...]}`. Note that we use string type for passage and query. You can also download our preprocessed data on huggingface hub: [official_train](https://huggingface.co/datasets/jacklin/msmarco_passage_ranking_corpus), [queries](https://huggingface.co/datasets/jacklin/msmarco_passage_ranking_queries) and [corpus](https://huggingface.co/datasets/jacklin/msmarco_passage_ranking_corpus).

## Training <a name="training"></a>
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

## Generate Passage and Query Embeddings <a name="generate_embeddings"></a>
```
export CUDA_VISIBLE_DEVICES=0
export MODEL=DHR #place DHR for DeLADE+[CLS] and DLR for DeLADE
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

## End-to-end Retrieval <a name="retrieval"></a>
### Retrieval on GPU <a name="retrieval_on_gpu"></a>
If you want to use GPU for retrieval, we suggest to use our implemented two-stage retrieval.
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

### Retrieval on CPU <a name="retrieval_on_cpu"></a>
If you only have CPU, we suggest to first quanize the index; then, use our implemented two-stage retrieval.
```
# index quanization
python -m retrieval.quantize_index \
--index_path ${INDEX_PATH}/index/${CORPUS}.index.pt \
--output_index_path ${INDEX_PATH}/index/${CORPUS}.pq64.faiss.index \
--qauntized_dim 64

# GIP retrieval
python -m retrieval.gip_retrieval \
--query_emb_path ${INDEX_PATH}/queries/queries.${CORPUS}.${SPLIT}.pt \
--index_path ${INDEX_PATH}/index/${CORPUS}.index.pt \
--faiss_pq_index_path ${INDEX_PATH}/index/${CORPUS}.pq64.faiss.index \
--emb_dim ${DLRDIM} \
--topk 1000 \
--lamda 1 \
--batch 1 \
--PQIP \
--rerank 
```

## Evaluation <a name="evaluation"></a>
The run file, result.trec, is in the trec format so that you can directly evaluate the result using pyserini.
```
python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank ${QREL_PATH} result.trec
python -m pyserini.eval.trec_eval -c -m recall.1000 ${QREL_PATH} result.trec
```


