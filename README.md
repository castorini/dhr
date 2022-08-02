# Dense Hybrid Retrieval
In this repo, we introduce two approaches to training transformers to capture semantic and lexical text representations for robust dense passage retrieval.
1. *[Aggretriever: A Simple Approach to Aggregate Textual Representation for Robust Dense Passage Retrieval](https://arxiv.org/abs/2208.00511)* Sheng-Chieh Lin, Minghan Li and Jimmy Lin. 
2. *[A Dense Representation Framework for Lexical and Semantic Matching](https://arxiv.org/abs/2206.09912)* Sheng-Chieh Lin and Jimmy Lin. 

This repo contains three parts: (1) densify (2) training (tevatron) (3) retrieval. 
Our training code is mainly from [Tevatron](https://github.com/texttron/tevatron) with a minor revision.

## Requirements
```
pip install torch>=1.7.0
pip install transformers==4.15.0
pip install pyserini
pip install beir
```
# Aggretriever

<img src="./fig/aggretriever.png" width="500">

In this paper, we introduce a simple approach to aggregating token-level information into a single-vector dense representation. We provide instruction for model training and evaluation on MS MARCO passage ranking dataset in the [document](https://github.com/castorini/dhr/blob/main/docs/aggretriever/msmarco-passage-train-eval.md). We also provide instruction for the evaluation on BEIR datasets in the [document](https://github.com/castorini/dhr/blob/main/docs/aggretriever/beir-eval.md).

# A Dense Representation Framework for Lexical and Semantic Matching
In this paper, we introduce a unified representation framework for Lexical and Semantic Matching. We first introduce how to use our framework to conduct retrieval for high-dimensional (lexcial) representations and combine with single-vector dense (semantic) representations for hybrid search.
## Dense Lexical Retrieval

<img src="./fig/densification.png" width="500">

We can densify any existing lexical matching models and conduct lexical matching on GPU. In the [document](https://github.com/jacklin64/DHR/blob/main/docs/densify_exp.md), we demonstrate how to conduct BM25 and uniCOIL end-to-end retrieval under our framework. Detailed description can be found in our [paper](https://arxiv.org/pdf/2112.04666.pdf). 

## Dense Hybrid Retrieval
With the densified lexical representations, we can easily conduct lexical and semantic hybrid retrieval using independent neural models. A document for hybrid retrieval will be coming soon.

## Dense Hybrid Representation Model

<img src="./fig/single_model_fusion.png" width="500">

In our paper, we propose a single model fusion approach by training the lexical and semantic components of a transformer while inference, we combine the densified lexical representations and dense representations as dense hybrid representations. Instead of training by yourself, you can also download our trained [DeLADE-CLS-P](https://huggingface.co/jacklin/DeLADE-CLS-P), [DeLADE-CLS](https://huggingface.co/jacklin/DeLADE-CLS) and [DeLADE](https://huggingface.co/jacklin/DeLADE) and directly peform inference on MSMARCO Passage dataset (see [document](https://github.com/jacklin64/DHR/blob/main/docs/dhr/msmarco-passage-train-eval.md)) or BEIR datasets (see [document](https://github.com/jacklin64/DHR/blob/main/docs/dhr/beir-eval.md)).
