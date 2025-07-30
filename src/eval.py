import json
import utils
from metrics import exact_presence
from metrics import get_asqa_metrics


def compute_metrics(result_file, dataset):
    res_data = []   # inference results
    with open(result_file, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            res_data.append(data)

    test_path = f"dataset/{dataset}/test.json"  # with ground truth
    test_data = utils.jload(test_path)[:len(res_data)]
    idx = 0
    num_accurate = 0
    n_docs = 5
    output_data_list = []
    for i, sample in enumerate(test_data):
        output = res_data[i]["predict"]
        if dataset == "ASQA":
            output_data = {
                "question": sample["question"],
                "answers": sample["answers"],
                "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
                "output": output,
                "ctxs": sample["ctxs"][:n_docs][::-1] if (
                            len(sample["ctxs"]) > 0 and sample["ctxs"][0]['score'] > sample["ctxs"][-1]['score']) else
                sample["ctxs"][::-1][:n_docs][::-1],
            }
            output_data_list.append(output_data)
        else:
            is_accurate = exact_presence(sample['answers'], output)
            idx += 1
            num_accurate += 1 if is_accurate else 0

    if dataset == "ASQA":
        get_asqa_metrics(output_data_list)
    else:
        accuracy = num_accurate / idx * 100
        print(f"Accuracy: {accuracy:.4f}%")


if __name__ == '__main__':
    result_file = 'results_nli_mismatch_rag_adaptive/generated_predictions.jsonl'
    dataset = "NaturalQuestions"   # NaturalQuestions, PopQA, TriviaQA, ASQA
    compute_metrics(result_file, dataset)
