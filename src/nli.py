from tqdm import tqdm
import utils
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def nli_annotate(tokenizer, nli_model, device, data_retrieval, data_statement):
    output_data_list = []
    for i, sample in tqdm(enumerate(data_retrieval), total=len(data_retrieval)):
        contexts = sample["ctxs"]
        hypothesis = data_statement[i]["qa_statement"]
        for idx in range(len(contexts)):
            premise = contexts[idx]["text"]

            with torch.no_grad():
                toks = tokenizer.encode(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation_strategy="only_first",
                ).to(device)
                logits = nli_model(toks)[0]

                # "contradiction" (dim 0), "neutral" (dim 1), "entailment" (2)
                entail_contradiction_logits = logits[:, [0, 1, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                contexts[idx]["nli_probs"] = [float(x) for x in probs[0]]
                contexts[idx]["entailment"] = 1 if max(contexts[idx]["nli_probs"]) == contexts[idx]["nli_probs"][-1] else 0

            torch.cuda.empty_cache()  # Clear GPU memory after processing
        sample["ctxs"] = contexts
        output_data_list.append(sample)
    return output_data_list


def data_filter(data, output_file):
    data_list = []
    for i, sample in enumerate(data):
        ctxs = sample["ctxs"]
        has_usable = False
        has_neg = False
        usable_list = []
        for idx, ctx in enumerate(ctxs):
            del sample["ctxs"][idx]["llm_label"]
            del sample["ctxs"][idx]["nli_probs"]
            if ctx["entailment"] == 1:
                sample["ctxs"][idx]["usable"] = True
                usable_list.append(idx)
                has_usable = True
            else:
                sample["ctxs"][idx]["usable"] = False
                has_neg = True
        if has_usable and has_neg:
            sample["usable_index"] = usable_list
            data_list.append(sample)

    utils.jdump(data_list, output_file)


if __name__ == '__main__':
    cache_dir = "facebook/bart-large-mnli"
    nli_model = AutoModelForSequenceClassification.from_pretrained(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    device = "cuda"
    nli_model.to(device)
    nli_model.eval()

    statement_path = '../dataset/NaturalQuestions/train_statement.json'
    data_statement = utils.jload(statement_path)

    retrieval_path = '../dataset/NaturalQuestions/train.json'
    data_retrieval = utils.jload(retrieval_path)[:len(data_statement)]

    labeled_data = nli_annotate(tokenizer, nli_model, device, data_retrieval, data_statement)
    output_file = "../dataset/NaturalQuestions/train_nli_label.json"
    data_filter(labeled_data, output_file)
