import torch, os, logging, random
import numpy as np
import json
import pandas as pd

random.seed(2021)
LANG_MAPs = {'EN': 'English',
             'FR': 'French',
             'DE': 'German',
             'ES': 'Spanish',
             'IT': 'Italian',
             'PL': 'Polish',
             'HU': 'Hungarian',
             'PT': 'Portuguese',
             'GR': 'Greek',
             'RO': 'Romanian',
             'RU': 'Russia',
             'SR': 'Serbian'}


def read_jsonl(filepath):
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            example = json.loads(line.strip())
            text_field = "text" if "text" in example else "content"
            if isinstance(example[text_field], str):
                examples.append(example)
            else:
                example.update({text_field: "none"})
                examples.append(example)
    return examples


def print_ex_results(json_obj, select_metric="Accuracy", table_shape=(5, 7)):
    res_dict = {}
    for seed, seed_res in json_obj.items():
        res = []
        for k, v in seed_res.items():
            if k != "config":
                res.append(v[select_metric])
        assert len(res) == table_shape[0] * table_shape[1]
        arr_res = np.array(res).reshape(table_shape)
        res = arr_res.transpose().tolist()
        res_dict[seed] = res

    for k, v in res_dict.items():
        print("-------------------")
        print(k)
        for line in v:
            print("\t".join([str(i) for i in line]))


def get_pos_neg(path, task="politics"):
    poses, negs = [], []
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line.strip())
            if ex["label"] == "positive":
                poses.append((ex["text"], f"{LANG_MAPs[ex['language']]} {task}"))
            else:
                negs.append((ex["text"], f"{LANG_MAPs[ex['language']]} normal"))
    return poses, negs


def split_out(ratio=0.1, filepath="/home/congcong/data_congcong/data/aisafe/personal50_aug2/train.json", random_seed=2021):
    random.seed(random_seed)
    examples = read_jsonl(filepath)
    label2examples = {}

    for example in examples:
        if example["label"] not in label2examples:
            label2examples[example["label"]] = []
        label2examples[example["label"]].append(example)

    label2count = {label: len(examples) for label, examples in label2examples.items()}
    print(f"**** Class dist. of original set ({os.path.dirname(filepath)}, seed={random_seed}) ({sum(label2count.values())}): {json.dumps(label2count, indent=2)}")

    samples = []
    for examples in label2examples.values():
        if isinstance(ratio, int):
            samples.extend(random.sample(examples, min(len(examples), ratio)))
        else:
            samples.extend(random.sample(examples, int(ratio * len(examples))))

    label2count = {label: int(ratio * len(examples)) if isinstance(ratio, float) else min(len(examples),ratio) for label, examples in label2examples.items()}
    print(f"**** Class dist. of down-sampled set ({os.path.dirname(filepath)}, ratio={ratio}, seed={random_seed}) ({sum(label2count.values())}): {json.dumps(label2count, indent=2)}")
    file_tag, file_ext = os.path.basename(filepath).split(".")[0], os.path.basename(filepath).split(".")[1]
    tf_name = f"{file_tag}_{str(ratio) if isinstance(ratio, float) else ratio}"
    tfile = f"{os.path.dirname(filepath)}/{tf_name}.{file_ext}"
    with open(tfile, "w+") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return tfile, tf_name


def prepare_gentit_data_np(filepath="/home/congcong/data_congcong/data/aisafe/personal50_aug2/train_0.1.json"):
    texts = []
    labels = []
    with open(filepath) as f:
        for line in f:
            ex = json.loads(line.strip())
            texts.append(ex["text"])
            labels.append(ex["label"])
    df = pd.DataFrame(columns=["text", "label"])
    df["text"] = texts
    df["label"] = labels

    file_tag, file_ext = os.path.basename(filepath).split(".")[0], os.path.basename(filepath).split(".")[1]
    genit_file_name = f"{file_tag}_genit"
    tfile = os.path.join(os.path.dirname(filepath), f"{genit_file_name}.csv")
    df.to_csv(tfile, index=False)
    return genit_file_name


def prepare_gentit_data(filepath="/home/congcong/data_congcong/data/aisafe/personal50_aug2/train_0.1.json", task="personal", include_tr=False, neg_nun=300):
    positives, negatives = [], []
    if include_tr:
        LANG_MAPs.update({"TR": "Turkish"})

    poses, negs = get_pos_neg(filepath, task=task)
    positives.extend(poses)
    negatives.extend(random.sample(negs, len(negs) if len(negs) >= neg_nun else neg_nun))

    examples = positives + negatives
    texts = []
    labels = []
    for each in examples:
        texts.append(each[0])
        labels.append(each[1])

    df = pd.DataFrame(columns=["text", "label"])
    df["text"] = texts
    df["label"] = labels

    file_tag, file_ext = os.path.basename(filepath).split(".")[0], os.path.basename(filepath).split(".")[1]
    genit_file_name = f"{file_tag}_genit"
    tfile = os.path.join(os.path.dirname(filepath), f"{genit_file_name}.csv")
    df.to_csv(tfile, index=False)
    return genit_file_name


def add_filehandler_for_logger(output_path, logger, out_name="log"):
    logFormatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(output_path, f"{out_name}.txt"), mode="a")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)


def get_pos_per_lang(datapath, filename=None):
    lang_tag = "EN"
    count = 0
    with open(os.path.join(datapath, "train.json" if filename is None else filename)) as f:
        for line in f:
            example = json.loads(line.strip())
            if example["label"] == "positive" and example["language"] == lang_tag:
                count += 1
    return count


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_class_names(filepath):
    classes = []
    with open(filepath, "r") as f:
        for line in f:
            classes.append(line.strip())
    return classes


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def cal_accuracy(preds, targets):
    return accuracy_score(targets, preds)


def cal_microprecision(preds, targets):
    return precision_score(targets, preds, average="micro")


def cal_microrecall(preds, targets):
    return recall_score(targets, preds, average="micro")


def cal_microf1(preds, targets):
    return f1_score(targets, preds, average='micro')


def cal_binaryprecision(preds, targets):
    return precision_score(targets, preds)


def cal_binaryrecall(preds, targets):
    return recall_score(targets, preds)


def cal_binaryf1(preds, targets):
    return f1_score(targets, preds)


def cal_macroprecision(preds, targets):
    return precision_score(targets, preds, average="macro")


def cal_macrorecall(preds, targets):
    return recall_score(targets, preds, average="macro")


def cal_macrof1(preds, targets):
    return f1_score(targets, preds, average='macro')


def cal_weightedf1(preds, targets):
    return f1_score(targets, preds, average='weighted')


def cal_weightedprecision(preds, targets):
    return precision_score(targets, preds, average="weighted")


def cal_weightedrecall(preds, targets):
    return recall_score(targets, preds, average="weighted")


METRICS2FN = {"Accuracy": cal_accuracy,
              "micro-F1": cal_microf1,
              "macro-F1": cal_macrof1,
              "weighted-F1": cal_weightedf1,
              "macro-Precision": cal_macroprecision,
              "macro-Recall": cal_macrorecall,
              "micro-Precision": cal_microprecision,
              "micro-Recall": cal_microrecall,
              "weighted-Precision": cal_weightedprecision,
              "weighted-Recall": cal_weightedrecall}


def calculate_perf(preds, targets):
    return_dict = {}
    if not isinstance(targets[0], list) and len(set(targets)) == 2:
        METRICS2FN.update({"binary-F1": cal_binaryf1, "binary-Recall": cal_binaryrecall, "binary-Precision": cal_binaryprecision})
    for k, v in METRICS2FN.items():
        return_dict[k] = round(v(preds, targets), 4)
    if isinstance(targets[0], list):
        return_dict["support"] = sum([sum(tgt) for tgt in targets])
    else:
        return_dict["support"] = len(targets)
    return return_dict
