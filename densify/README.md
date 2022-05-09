# Densify Sparse Vector
The repo is to demonstrate how to densify existing sparse lexical retrievers for dense search. We use [pyserini](https://github.com/castorini/pyserini) to get the sparse vectors from models. We show how to densify BM25 and uniCOIL on msmarco-passage ranking dataset in this repo.   

# Densifying BM25
## Data Prepare
Folloing the [instruction](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md), we first download MSMARCO passage and query collection files in ${COLLECTION_PATH} and ${Q_PATH}, and convert the collection.tsv into json file for pyserini index.
## Output BM25 Vector from index
We first index the corpus using BM25.
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
We now start to densify corpus and queries.
```shell=bash
export DLRDIM=768
export CORPUS=msmarco-passage
export DLR_PATH=BM25_DIM${DLRDIM}

python -m densify.densify_corpus \
    --prefix ${CORPUS} \
    --index_path ${INDEX_PATH} \
    --vector_dir ${VECTOR_DIR} \
    --output_dir ${DLR_PATH} \
    --output_dims ${DLRDIM}

python -m densify.densify_query \
    --prefix ${CORPUS} \
    --index_path ${INDEX_PATH} \
    --query_path ${Q_PATH} \
    --output_dir ${DLR_PATH} \
    --output_dims ${DLRDIM} \
    --analyze 
```

We then merge index and start DLR search.
```shell=bash
# Merge index
export DLRDIM=768
export CORPUS=msmarco-passage
export DLR_PATH=BM25_DIM${DLRDIM}
export SPLIT=dev.small

python -m retrieval.index \
--index_path ${DLR_PATH} \
--index_prefix ${CORPUS} \
--index
mkdir ${DLR_PATH}/index
mv ${DLR_PATH}/${CORPUS}.index.pt ${DLR_PATH}/index/

python -m retrieval.gip_retrieval \
  --query_emb_path ${DLR_PATH}/queries/queries.${CORPUS}.${SPLIT}.pt \
  --emb_dim ${DLRDIM} \
  --index_path ${DLR_PATH}/index/${CORPUS}.index.pt \
  --M 1 \
  --rerank \
  --use_gpu \
  --lamda 1
```