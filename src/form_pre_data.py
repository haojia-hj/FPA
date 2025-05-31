import os
import sys
import argparse
import utils
from metrics import exact_presence
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import random
import copy


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    if question.startswith("."):
        question = question.lstrip(". ")

    return question[0].lower() + question[1:]


def format_prompt(data: dict,
    tokenizer: PreTrainedTokenizer,
    n_docs: int
):
    example = copy.deepcopy(data)

    formatted_data = []
    max_length = tokenizer.model_max_length
    prefix = "<|start_header_id|>user<|end_header_id|>\n\n"
    target_prefix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prefix_tokenized_id = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_len = len(prefix_tokenized_id)

    ctxs = example["ctxs"]
    ctxs = sorted(ctxs, key=lambda x: x["score"], reverse=True)
    usable_index = example["usable_index"]
    new_ctxs = []
    new_usable = []
    new_neg = []

    for idx in usable_index[::-1]:
        new_ctxs.append(ctxs[idx])
        ctxs.pop(idx)
    if len(new_ctxs) > n_docs-1:
        new_ctxs = new_ctxs[:n_docs-1]
    neg_index = random.sample(range(len(ctxs)), n_docs-len(new_ctxs))
    for idx in neg_index:
        new_ctxs.append(ctxs[idx])
    random.shuffle(new_ctxs)
    for idx, ctx in enumerate(new_ctxs):
        if ctx["usable"]:
            new_usable.append(idx)
        else:
            new_neg.append(idx)

    question = normalize_question(example["question"])
    usable_doc = ", ".join(f"{idx + 1}" for idx in new_usable)
    neg_doc = ", ".join(f"{idx + 1}" for idx in new_neg)
    if len(new_usable) > 2:
        ref_doc1 = ", ".join(f"{idx + 1}" for idx in new_usable[:-1]) + ", and " + str(new_usable[-1]+1)
        ref_doc1 = "\nAccording to documents " + ref_doc1 + ", "
    elif len(new_usable) == 2:
        ref_doc1 = ", ".join(f"{idx + 1}" for idx in new_usable[:-1]) + " and " + str(new_usable[-1] + 1)
        ref_doc1 = "\nAccording to documents " + ref_doc1 + ", "
    else:
        ref_doc1 = f"\nAccording to document {new_usable[0] + 1}, "
    ref_doc2 = [f"\nAccording to document {idx + 1}, " for idx in new_neg]
    ref_list = [ref_doc1] + ref_doc2 + [""]
    for ref_doc in ref_list:
        query = f"Based on your knowledge and the provided information, answer the question:\n{question}"
        docs_text = "\n\n".join(
            [f"Document {idx + 1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(new_ctxs)])
        docs_text = f"{docs_text}\n\n"
        input_ids = tokenizer(docs_text + query + ref_doc + target_prefix, return_tensors="pt", add_special_tokens=False).input_ids

        if input_ids.shape[-1] > max_length - prefix_len:
            input_ids = input_ids[..., -(max_length - prefix_len):]
        input_ids = torch.cat([prefix_tokenized_id, input_ids], axis=-1)

        formatted_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        formatted_data.append(formatted_prompt)

    return formatted_data[:-1], formatted_data[-1], usable_doc, neg_doc, ref_list[:-1]


def generate_response(args):
    data_path = f'dataset/{args.dataset_name}/train_nli_label.json'
    print(f"Loading dataset: {data_path}")
    train_data = utils.jload(data_path)[:args.max_instances]

    print(f'Loading model: {args.model_dir}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    output_data_list = []
    for sample in tqdm(train_data):
        prompts, instruction, usable_doc, neg_doc, ref_docs = format_prompt(
            data=sample,
            tokenizer=tokenizer,
            n_docs=args.n_docs
        )
        output_text_list = []
        chosen = ""
        rejected = ""
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
            ref_doc = ref_docs[i]
            trig_input_ids = tokenizer(ref_doc, return_tensors="pt").to(model.device).input_ids
            trig_tokens = len(trig_input_ids[0])

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            attention_mask = tokenizer(prompt, return_tensors="pt").to(model.device).attention_mask

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens-trig_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=1
            )
            response = outputs[0][input_ids.shape[-1]:]
            output_text = tokenizer.decode(response, skip_special_tokens=True)
            output_text_list.append(output_text)

        for i, output in enumerate(output_text_list):
            is_accurate = exact_presence(sample['answers'], output)
            if i == 0 and is_accurate:
                chosen = output
            elif i != 0 and not is_accurate:
                rejected = output

            if chosen and rejected:
                output_data = {
                    "instruction": instruction,
                    "input": None,
                    "chosen": chosen,
                    "usable_doc": usable_doc,
                    "rejected": rejected,
                    "neg_doc": neg_doc
                    }
                output_data_list.append(output_data)
                rejected = ""

    train_file = os.path.join(args.output_dir, "dpo_train_nq_4096.json")
    utils.jdump(output_data_list, train_file)
    print(f"Outputs saved to {train_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='NaturalQuestions', help='Name of the dataset')
    parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved documents')
    parser.add_argument('--output_dir', type=str, help='Path to the output file')
    parser.add_argument('--model_dir', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='Directory to cached models or name of the model in Hugging Face model hub')
    parser.add_argument('--max_new_tokens', type=int, default=4096, help='Maximum number of tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_instances', type=int, default=sys.maxsize, help='Maximum number of data samples')

    args = parser.parse_args()

    random.seed(args.seed)
    generate_response(args)

