## All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment

This is the official implementation for paper _All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment_.

## Content

- [Installation](#installation)
- [Retrieve](#retrieve)
  - [Download data](#download-data)
  - [Retrieve with your own data](#retrieve-with-your-own-data)
- [Construct Training Dataset](#construct-training-dataset)
- [Training & Inference](#training--inference)
- [Evaluation](#evaluation)

## Installation

To install dependent Python libraries, run

```
conda create -n fpa python=3.10 -y
conda activate fpa
pip install -r requirements.txt
```

## Retrieve

### Download data

You can download all QA datasets with retrieved results used in our work from this [link](https://drive.google.com/file/d/18IP07_r-59JV6Nn5LlycIxEPSq4Nismm/view?usp=share_link)
or by running this command:

```
gdown https://drive.google.com/uc?id=18IP07_r-59JV6Nn5LlycIxEPSq4Nismm
```

### Retrieve with your own data

In our work, we use different retrievers (including DPR, Contriever MSMARCO, and GTR) to retrieve documents.

For [DPR](https://github.com/facebookresearch/DPR), you can first install the DPR repo, then download Wikipedia passages,
passage embeddings and a biencoder checkpoint.

```
python dpr/data/download_data.py --resource data.wikipedia_split.psgs_w100
python dpr/data/download_data.py --resource data.retriever_results.nq.single.wikipedia_passages
python dpr/data/download_data.py --resource checkpoint.retriever.single.nq.bert-base-encoder
```

You can run the script below to get the retrieval results.

```
python dense_retriever.py \
	model_file={path to a checkpoint file} \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=["~/myproject/downloads/data/retriever_results/nq/single/wikipedia_passages_*"] \
	out_file={path to output json file with results} 
```

Alternatively, you can directly download the top-100 DPR retrieval results for NQ from these links:
 [nq-train](https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-train.json.gz),
 [nq-dev](https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz) and
 [nq-test](https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz), refer to
 [here](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py).

For [Contriever-MSMARCO](https://github.com/facebookresearch/contriever), you can first complete setup, then download
Wikipedia passages and passage embeddings pre-computed with Contriever-MSMARCO.
```
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
```
You can run the following command to retrieve top-5 passages.
```
python passage_retrieval.py 
    --model_name_or_path facebook/contriever-msmarco \
    --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data path/to/YOUR_INPUT_DATA.json \
    --output_dir path/to/YOUR_OUTPUT_DIR \
    --n_docs 5 \
    --no_fp16
```

For GTR, please refer to this [repo](https://github.com/google-research/t5x_retrieval). We use the top-100 GTR retrieval
results for ASQA provided by ALCE. You can install the [ALCE repo](https://github.com/princeton-nlp/ALCE?tab=readme-ov-file#data)
and directly download the data.

## Construct Training Dataset

The fact-centric positive document mining method determines the relevance of documents by checking whether the facts 
contained in a given document are consistent with the ground truth facts. For short-form QA datasets such as NQ, 
where the answer is not a complete statement of fact, you can run the script to convert the QA pairs into statements
of fact using a LLM.

```
bash gen_qa.sh
```

We use a NLI model to predict whether a document and a ground truth statement are entailment. Consider the document as
_premise_ and the ground truth statement as _hypothesis_; if the two texts are entailment, they are deemed to be 
factual consistent. You can run NLI-based annotation by running the command below.

```
python src/nli.py
```

The documents that consist with the ground truth facts are classified as positive documents, while the other documents 
are classified as negative documents. We use the obtained positive and negative documents to construct the training data
by automatically sampling LLM's responses. By adding a trigger with document indexes (e.g., _“According to documents 1 
and 5”_) at the end of the prompt, it is possible to control which documents are referenced when sampling the chosen
and rejected responses. You can use the following script to construct preference dataset.

```
bash form_pre_data.sh
```

Run the following script to employ citation mismatch data augmentation strategy.

```
python src/mismatch_data_aug.py
```

## Training & Inference

Our fine-tuning and inference are implemented on the basis of [Llama Factory](https://github.com/hiyouga/LLaMA-Factory).
If you want to use this framework, you can first complete the installation of Llama Factory and the preparation of the 
dataset. Specifically, please copy the training dataset constructed in the previous step to the `LLaMA-Factory/data` 
directory, and add the custom dataset in `LLaMA-Factory/data/dataset_info.json`, for example

```
"dpo_train_nq_4096_mismatch": {
    "file_name": "dpo_train_nq_4096_mismatch.json",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
}
```

See [config/train_model.json](https://github.com/haojia-hj/FPA/blob/main/config/train_model.json) for our fine-tuning settings.
You can copy the `train_model.json` file to the `Llama-Factory/` directory and use the following command to run LoRA fine-tuning.

```
llamafactory-cli train train_model.json
```

If you want to implement inference using Llama Factory, you can first run the following command to convert the dataset 
into the appropriate format.

```
python src/format_test_data.py
```

Please copy the formatted dataset to the `LLaMA-Factory/data` directory and update `LLaMA-Factory/data/dataset_info.json`, for example

```
"test_nq_rag_adaptive": {
    "file_name": "test_nq_rag_adaptive.json",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
}
```

Then you can use the config file [config/infer.json](https://github.com/haojia-hj/FPA/blob/main/config/infer.json) to run inference.

Alternatively, you can use `src/inference.py` to run inference by using the following script without data formatting.

```
bash infer.sh
```

But please first run the merging of the LoRA fine-tuning model using Llama Factory.

## Evaluation

For NQ, PopQA and TriviaQA, we evaluate the accuracy based on whether the output contains ground truth answers.
For ASQA, we use the official metrics in ALCE, including correctness (str-em), citation precision (pre) and recall (rec).
You can run the following script to calculate accuracy and correctness (str-em). The implementation of the calculation for citation 
precision and recall refers to [ALCE](https://github.com/princeton-nlp/ALCE?tab=readme-ov-file#evaluation).

```
python src/eval.py
```