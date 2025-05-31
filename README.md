## All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment

This is the official implementation for paper _All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment_.

## Installation

To install dependent Python libraries, run

```
conda create -n fpa python=3.10 -y
conda activate fpa
pip install -r requirements.txt
```

## Retrieve

### Download data


### Retrieve with your own data


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

## Training


## Inference


## Evaluation
