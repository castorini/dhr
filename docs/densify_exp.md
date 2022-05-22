# Densify Sparse Vector
The repo is to demonstrate how to densify existing sparse lexical retrievers for dense search. We use [pyserini](https://github.com/castorini/pyserini) to get the sparse vectors from models. We show how to densify BM25 on msmarco-passage ranking dataset in this repo.   

# Densifying BM25
## Data Prepare
Folloing the [instruction](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md), we first download MSMARCO passage collection and query files. Then, convert the collection.tsv into json file in $COLLECTION_PATH for pyserini index, and put queries.dev.small.tsv file into $Q_DIR.
```shell=bash
export COLLECTION_PATH=need_your_assignment
export INDEX_PATH=need_your_assignment
export VECTOR_DIR=need_your_assignment
export Q_DIR=need_your_assignment
export MODEL=BM25
export DLRDIM=768
export CORPUS=msmarco-passage
export DLR_PATH=${MODEL}_DIM${DLRDIM}
export SPLIT=dev.small
```
## Output BM25 Vector from index
We first index the json corpus using BM25.
```shell=bash
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${COLLECTION_PATH} \
  --index ${INDEX_PATH} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 12 \
  --storeDocvectors --storeRaw --optimize
```
Then, we output the sparse vector in a json file. We split the json file into multiple splits for multi-process in the next step.
```shell=bash
python -m densify.output_vector \
  --index_path ${INDEX_PATH} \
  --output_path ${VECTOR_DIR}/split.json

split -a 2 -dl 1000000 --additional-suffix=.json ${VECTOR_DIR}/split.json ${VECTOR_DIR}/split 
rm ${VECTOR_DIR}/split.json
```
## Sparse vector densification
We now start to densify corpus and queries.
```shell=bash
python -m densify.densify_corpus \
    --model ${MODEL} \
    --prefix ${CORPUS} \
    --tokenizer ${INDEX_PATH} \
    --vector_dir ${VECTOR_DIR} \
    --output_dir ${DLR_PATH} \
    --output_dims ${DLRDIM}

python -m densify.densify_query \
    --model bm25 \
    --prefix ${CORPUS} \
    --tokenizer ${INDEX_PATH} \
    --query_path ${Q_DIR}/queries.${SPLIT}.tsv \ \
    --output_dir ${DLR_PATH} \
    --output_dims ${DLRDIM} \
```
## BM25 search on GPU
We then merge index and start DLR search.
```shell=bash
# Merge index
python -m retrieval.index \
--index_path ${DLR_PATH}/encoding \
--index_prefix ${CORPUS} \

mkdir ${DLR_PATH}/encoding/index
mv ${DLR_PATH}/encoding/${CORPUS}.index.pt ${DLR_PATH}/encoding/index/

# Search
python -m retrieval.gip_retrieval \
  --query_emb_path ${DLR_PATH}/encoding/queries/queries.${CORPUS}.${SPLIT}.pt \
  --emb_dim ${DLRDIM} \
  --index_path ${DLR_PATH}/encoding/index/${CORPUS}.index.pt \
  --theta 1 \
  --rerank \
  --use_gpu \
```

# Densifying uniCOIL
## Data Prepare
Folloing the [instruction](https://github.com/castorini/pyserini/blob/master/docs/experiments-unicoil.md), we download pre-encoded uniCOIL passage collection. 
```shell=bash
wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/data/msmarco-passage-unicoil.tar -P collections/

tar xvf collections/msmarco-passage-unicoil.tar -C collections/
```
```shell=bash
export MODEL=uniCOIL
export DLRDIM=768
export CORPUS=msmarco-passage
export VECTOR_DIR=./collections/msmarco-passage-unicoil-b8
export DLR_PATH=${MODEL}_DIM${DLRDIM}
export SPLIT=dev.small
```
## Sparse vector densification
We now start to densify corpus and queries.
```shell=bash
python -m densify.densify_corpus \
    --model ${MODEL} \
    --prefix ${CORPUS} \
    --vector_dir ${VECTOR_DIR} \
    --output_dir ${DLR_PATH} \
    --output_dims ${DLRDIM}

python -m densify.densify_query \
    --model ${MODEL} \
    --prefix ${CORPUS} \
    --query_path ${Q_DIR}/queries.${SPLIT}.tsv \ \
    --output_dir ${DLR_PATH} \
    --output_dims ${DLRDIM} \
```

We then merge index and start DLR search.
```shell=bash
# Merge index
python -m retrieval.index \
--index_path ${DLR_PATH}/encoding \
--index_prefix ${CORPUS} \

mkdir ${DLR_PATH}/encoding/index
mv ${DLR_PATH}/encoding/${CORPUS}.index.pt ${DLR_PATH}/encoding/index/

# Search
python -m retrieval.gip_retrieval \
  --query_emb_path ${DLR_PATH}/encoding/queries/queries.${CORPUS}.${SPLIT}.pt \
  --emb_dim ${DLRDIM} \
  --index_path ${DLR_PATH}/encoding/index/${CORPUS}.index.pt \
  --theta 1 \
  --rerank \
  --use_gpu \
```