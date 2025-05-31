import os
import sys
import argparse
import data_process
import utils
from metrics import exact_presence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json


def generate_qa_statement(args):
    data_path = f'dataset/{args.dataset_name}/train.json'
    print(f"Loading dataset: {data_path}")
    train_data = utils.jload(data_path)[:args.max_instances]

    print(f'Loading model: {args.model_dir}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    prompt_dict = utils.jload(args.prompt_dict_path)
    prompts = data_process.format_prompt_with_data_list(
        data_list=train_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        do_qa_statement_generation=True,
    )

    output_text_list = []
    for prompt in tqdm(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        attention_mask = tokenizer(prompt, return_tensors="pt").to(model.device).attention_mask

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        response = outputs[0][input_ids.shape[-1]:]
        output_text = tokenizer.decode(response, skip_special_tokens=True)
        output_text_list.append(output_text)

    eval_result = save_outputs(output_text_list, train_data, args)


def infer(args):
    data_path = f'dataset/{args.dataset_name}/test.json'
    print(f"Loading dataset: {data_path}")
    test_data = utils.jload(data_path)[:args.max_instances]

    print(f'Loading model {args.model_dir}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    prompt_dict = utils.jload(args.prompt_dict_path)
    prompts = data_process.format_prompt_with_data_list(
        data_list=test_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
    )
    print(model.device)

    output_text_list = []
    for prompt in tqdm(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        attention_mask = tokenizer(prompt, return_tensors="pt").to(model.device).attention_mask

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=1
        )
        response = outputs[0][input_ids.shape[-1]:]
        output_text = tokenizer.decode(response, skip_special_tokens=True)
        output_text_list.append(output_text)

    eval_result = save_outputs(output_text_list, test_data, args)
    print(eval_result)


def save_outputs(outputs, test_data, args):
    output_data_list = []
    idx = 0
    num_accurate = 0
    for i, output in enumerate(outputs):
        sample = test_data[i]
        if args.do_qa_statement_generation:
            sample["qa_statement"] = output.replace("Statement: ", "").replace("Here is the rewritten statement of fact: ", "").replace("Here is the rewritten statement of fact:\n\n", "").strip()
            output_data_list.append(sample)
        else:
            is_accurate = exact_presence(sample['answers'], output)
            output_data = {
                "question": sample["question"],
                "answers": sample["answers"],
                "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
                "ctxs": sample["ctxs"][:args.n_docs][::-1] if (len(sample["ctxs"]) > 0 and sample["ctxs"][0]['score'] > sample["ctxs"][-1]['score']) else sample["ctxs"][::-1][:args.n_docs][::-1],
                "output": output,
                "is_accurate": 1 if is_accurate else 0
                }
            output_data_list.append(output_data)
            idx += 1
            num_accurate += 1 if is_accurate else 0

    if args.do_qa_statement_generation:
        output_file = f'dataset/{args.dataset_name}/train_statement.json'
        utils.jdump(output_data_list, output_file)
        print(f"Outputs saved to {output_file}")
        eval_result = None
    else:
        output_file = os.path.join(args.output_dir, "result_test.json")
        utils.jdump(output_data_list, output_file)
        print(f"Outputs saved to {output_file}")

        accuracy = num_accurate / idx * 100
        print(f"Accuracy: {accuracy:.1f}%")
        eval_result = {"accuracy": accuracy, "num_examples": idx}

        with open(f"{args.output_dir}/metrics_test.json", "w") as f:
            f.write(json.dumps(eval_result) + "\n")

    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--do_qa_statement_generation', action='store_true', help='Generate qa statements on training data')
    parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved documents')
    parser.add_argument('--output_dir', type=str, help='Path to the output file')
    parser.add_argument('--model_dir', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Directory to cached models or name of the model in Hugging Face model hub')
    parser.add_argument('--prompt_dict_path', type=str, default="src/rag.json")
    parser.add_argument('--max_new_tokens', type=int, default=15, help='Maximum number of tokens')
    parser.add_argument('--max_instances', type=int, default=sys.maxsize, help='Maximum number of data samples')

    args = parser.parse_args()

    if args.do_qa_statement_generation:
        generate_qa_statement(args)
    else:
        infer(args)
