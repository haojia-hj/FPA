import utils
import random


def mismatch_aug(data):
    data_list = []
    for i, sample in enumerate(data):
        chosen = sample["chosen"]
        usable_doc = eval("[" + sample["usable_doc"] + "]")
        neg_doc = eval("[" + sample["neg_doc"] + "]")
        rand_num = random.randint(1, len(neg_doc))
        rand_idx = random.sample(neg_doc, rand_num)
        mis_idx = sorted(rand_idx)
        if len(mis_idx) > 2:
            ref_doc = ", ".join(f"{idx}" for idx in mis_idx[:-1]) + ", and " + str(mis_idx[-1])
            ref_doc = "According to documents " + ref_doc + ", "
        elif len(mis_idx) == 2:
            ref_doc = ", ".join(f"{idx}" for idx in mis_idx[:-1]) + " and " + str(mis_idx[-1])
            ref_doc = "According to documents " + ref_doc + ", "
        else:
            ref_doc = f"According to Document {mis_idx[0]}, "

        if len(usable_doc) < 3 or "provided documents" in chosen:
            chosen_text = chosen.split(", ", 1)[1]
        else:
            chosen_text = chosen.split(", ", len(usable_doc))[-1]
        mis_reject = ref_doc + chosen_text
        new_data = {
            "instruction": sample["instruction"],
            "input": sample["input"],
            "chosen": sample["chosen"],
            "usable_doc": sample["usable_doc"],
            "rejected": mis_reject,
            "neg_doc": sample["neg_doc"]
        }
        data_list.append(new_data)
    return data_list


if __name__ == '__main__':
    data_path = "dataset/NaturalQuestions/dpo_train_nq_4096.json"
    data = utils.jload(data_path)
    data_list = mismatch_aug(data)
    data_list = list(set(data_list))
    total_data = data + data_list
    random.seed(42)
    random.shuffle(total_data)
    output_file = "dataset/NaturalQuestions/dpo_train_nq_4096_mismatch.json"
    utils.jdump(total_data, output_file)