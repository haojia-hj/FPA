## All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment

This is the official implementation for paper _All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment_.

## Installation

To install dependent Python libraries, run

```
pip install -r requirements.txt
```

Alternatively, create a conda environment

```
conda env create -f environment.yml
```

## Retrieve

### Download data


### Retrieve with your own data


## Construct Training Dataset

Convert QA pairs into statements of fact

```
bash gen_qa.sh
```

NLI-based annotation

```
python src/nli.py
```

construct preference dataset

```
bash form_pre_data.sh
```

citation mismatch data augmentation

```
python src/mismatch_data_aug.py
```

## Training


## Inference


## Evaluation
