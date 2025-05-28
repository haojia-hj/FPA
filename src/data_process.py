# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import json
from tqdm import tqdm
from functools import partial
from typing import Dict, Sequence, Union

import torch
import numpy as np
import transformers
import common_utils


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    if question.startswith("."):
        question = question.lstrip(". ")

    return question[0].lower() + question[1:]


def build_contexts(example, n_docs):
    if len(example["ctxs"]) > 0 and example["ctxs"][0]["score"] > example["ctxs"][-1]["score"]:
        ctxs_list = example["ctxs"][:n_docs][::-1]
    else:
        ctxs_list = example["ctxs"][::-1][:n_docs][::-1]

    docs_text = "\n\n".join([f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(ctxs_list)])
    doc_prompt = f"{docs_text}\n\n"
    
    return doc_prompt


def format_prompt(
        dataset_name: str,
        example: dict, 
        n_docs: int,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        do_qa_statement_generation: bool,
        ) -> str:
    """Formats a prompt with a prompt_dict formatter.

    Args:
        example: A dict-like object with required keys "instruction" and "input"
        prompt_dict: Dictionary containing the keys "prompt_noinputs" and "prompt_inputs" which have
            placeholders corresponding to the keys from `example`. E.g. "{instruction}".

    Returns:
        A formatted prompt string.

    Examples
    --------
    >>> format_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    example['question'] = normalize_question(example['question'])
    if do_qa_statement_generation:
        example['answers'] = "\n".join(example['answers'])
    max_length = tokenizer.model_max_length

    query_prompt = prompt_dict['query_prompt'].format_map(example)
    target_prefix = ""

    doc_prompt = build_contexts(example, n_docs=n_docs)

    prefix = prompt_dict['user_prefix']

    prefix_tokenized_id = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_len = len(prefix_tokenized_id)

    target_prefix += prompt_dict['assistant_prefix']

    input_ids = tokenizer(doc_prompt + query_prompt + target_prefix, return_tensors="pt", add_special_tokens=False).input_ids

    if input_ids.shape[-1] > max_length - prefix_len:
        input_ids = input_ids[..., -(max_length - prefix_len):]
    input_ids = torch.cat([prefix_tokenized_id, input_ids], axis=-1)
    
    formatted_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return formatted_prompt


def format_prompt_with_data_list(
    data_list: list[dict],
    dataset_name: str,
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    n_docs: int = 5,
    do_qa_statement_generation: bool = False
):

    data = copy.deepcopy(data_list)
    formatted_data = [format_prompt(dataset_name, example, n_docs, prompt_dict, tokenizer, do_qa_statement_generation) for example in tqdm(data)]

    return formatted_data
