# BEIR Evaluation
We provide two scripts for BEIR evaluation and use the model, [DeLADE-CLS-P](https://huggingface.co/jacklin/DeLADE-CLS-P), and the dataset, trec-covid, as an example.
## Evaluation with GIP
We first downlaod our model and beir dataset.
```
git clone https://huggingface.co/jacklin/DeLADE-CLS-P
export MODEL_DIR=DeLADE-CLS-P
export CORPUS=trec-covid
export SPLIT=test
python -m tevatron.datasets.beir.preprocess --dataset ${CORPUS}
```
Then we tokenize the query and corpus.
```
python -m tevatron.utils.tokenize_corpus \
  --corpus_path ./dataset/${CORPUS}/corpus/collection.json \
  --output_dir ./dataset/${CORPUS}/tokenized_data/corpus \
  --corpus_domain beir \
  --tokenize --encode --num_workers 10

python -m tevatron.utils.tokenize_query \
  --qry_file ./dataset/${CORPUS}/queries/queries.${SPLIT}.tsv \
  --output_dir ./dataset/${CORPUS}/tokenized_data/queries

```
Following the [inference scripts](https://github.com/castorini/DHR/blob/main/docs/msmarco-passage-train-eval.md#inference-msmarco-passage-for-retrieval) for msmarco-passage data, we run inference, GIP retrieval and evaluation on the BEIR dataset.
```
export CUDA_VISIBLE_DEVICES=0
export MODEL=DHR
export CLSDIM=128
export DLRDIM=768
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
The second one is to directly use [BEIR](https://github.com/beir-cellar/beir) API to conduct brute-force search. No densification before retrieval; thus, the result is slightly different from the numbers reported in our paper. Note that, for this script, we currently only support our DHR models, [DeLADE-CLS](https://huggingface.co/jacklin/DeLADE-CLS) and [DeLADE-CLS-P](https://huggingface.co/jacklin/DeLADE-CLS-P). 
```
python -m tevatron.datasets.beir.encode_and_retrieval --dataset trec-covid --model ${MODEL_DIR}
```



