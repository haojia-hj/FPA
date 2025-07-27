import utils


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    if question.startswith("."):
        question = question.lstrip(". ")

    return question[0].lower() + question[1:]


def format_test_dataset(dataset_name, data, n_docs, adaptive_retrieval):
    task_type = ""
    data_list = []
    for i, sample in enumerate(data):
        question = normalize_question(sample["question"])
        if dataset_name == "ASQA":
            answers = sample["qa_pairs"][0]["answers"][0]
        else:
            answers = sample["answers"][0]

        if n_docs == 0:
            instruction = f"Based on your knowledge, answer the question:\n{question}"
            task_type = "direct"
        else:
            if len(sample["ctxs"]) > 0 and sample["ctxs"][0]["score"] > sample["ctxs"][-1]["score"]:
                ctxs_list = sample["ctxs"][:n_docs][::-1]
            else:
                ctxs_list = sample["ctxs"][::-1][:n_docs][::-1]

            doc_text = "\n\n".join([f"Document {idx + 1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(ctxs_list)])

            instruction = f"{doc_text}\n\nBased on your knowledge and the provided information, answer the question:\n{question}"
            task_type = "rag"
            if adaptive_retrieval:
                task_type += "_adaptive"
                instruction += " If the provided documents are irrelevant or do not contain complete information to answer the question, supplement and answer with your knowledge."

        if dataset_name == "ASQA":
            instruction = "Answer the given question and cite the index of all references used in your answer. Ensure both the answer and citations are accurate.\n\n" + instruction

        test_data = {
            "instruction": instruction,
            "input": None,
            "output": answers,
            "system": None,
            "history": []
        }
        data_list.append(test_data)

    output_file = f"dataset/{dataset_name}/test_{task_type}.json"
    utils.jdump(data_list, output_file)


if __name__ == '__main__':
    dataset_name = "PopQA"
    data_path = f"dataset/{dataset_name}/test.json"
    data = utils.jload(data_path)
    format_test_dataset(dataset_name, data, n_docs=5, adaptive_retrieval=True)
