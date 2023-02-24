from transformers import get_constant_schedule, get_constant_schedule_with_warmup, AdamW, get_linear_schedule_with_warmup, AutoConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm, trange
from sklearn.metrics import classification_report

from src.encoder import EncoderArgs
from src.datasets import MyDataset

from src.utils import *
import json

import transformers
from torch.utils.data import DataLoader
import pandas as pd
import time
from torch.nn.parallel import DataParallel

def soft_label(input_tensor):
    '''soft labeling, following: https://arxiv.org/pdf/2010.07245.pdf'''
    # empirically, works slightly better than without this
    weight = input_tensor ** 2 / torch.sum(input_tensor, dim=0)
    target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
    # equals
    # target_dist = (weight / torch.sum(weight, dim=1).view(-1,1))
    return target_dist


class EncoderTrainer():
    def __init__(self, args: EncoderArgs):
        transformers.logging.set_verbosity_info()
        self.logger = transformers.logging.get_logger()
        self.args = args
        self.raw_classes = read_class_names(os.path.join(self.args.data_path, "class_names.txt"))
        add_filehandler_for_logger(os.path.join(self.args.data_path, "logs"), self.logger, out_name="log_" + self.__class__.__name__)
        self.logger.info("General Data Args: " + json.dumps(self.args.__dict__, indent=2))
        self.logger.info("General Training Args: " + json.dumps(self.args.__dict__, indent=2))
        set_seed(self.args.seed)
        self.start_time = time.time()
        self.device_count = torch.cuda.device_count()

    def report_train_important_params(self, **other_args):
        important_params = {}
        model_short_name = self.args.model_name_or_path.split('/')[-1]
        important_params.update({"max_seq_length": self.args.max_seq_length})
        important_params.update({"base_model": model_short_name})
        important_params.update({"train_epochs": self.args.train_epochs})
        important_params.update({"train_batch_size_per_device": self.args.train_batch_size_per_device})
        important_params.update({"device(gpu)_count": self.device_count})
        important_params.update({"learning_rate": self.args.train_training_lr})
        important_params.update({"seed": self.args.seed})
        important_params.update({"warmup_ratio": self.args.warmup_ratio})
        important_params.update({"weight_decay": self.args.weight_decay})
        important_params.update({"lr_scheduler": self.args.train_lr_scheduler})
        important_params.update({"other_info": other_args})
        self.logger.info("Important params: " + json.dumps(important_params, indent=2))

    def encode_data(self, tokenizer, examples, with_label=False,soft_labels=False,nla=False):
        encoded_examples = {}
        text_field = "text" if "text" in examples[0] else "content"
        for i in trange(0, len(examples), self.args.encoding_batch_size):
            batch_examples = examples[i:i + self.args.encoding_batch_size]
            input_texts = [be[text_field].lower() for be in batch_examples]
            inputs = tokenizer(input_texts, padding="max_length", return_tensors="pt", truncation=True, max_length=self.args.max_seq_length)
            if with_label:
                if self.args.multi_label:
                    one_hots = []
                    for be in batch_examples:
                        labels_this = be["label"].split(",")
                        one_hot = [1 if i in labels_this else 0 for i in self.raw_classes]
                        one_hots.append(one_hot)
                    inputs.update({"labels": torch.tensor(one_hots).float()})
                else:
                    inputs.update({"labels": [self.raw_classes.index(be["label"]) for be in batch_examples]})

                if soft_labels:
                    inputs.update({"probs": [be["probs"] for be in batch_examples]})
            if nla:
                inputs.update({"is_aug": [int(be["aug"]) if "aug" in be else 0 for be in batch_examples]})

            for k, v in inputs.items():
                if k not in encoded_examples:
                    encoded_examples[k] = v
                else:
                    if torch.is_tensor(v):
                        encoded_examples[k] = torch.cat([encoded_examples[k], v], dim=0)
                    else:
                        encoded_examples[k].extend(v)

        if soft_labels:
            encoded_examples["probs"]=torch.tensor(encoded_examples["probs"] )
                # encoded_examples["probs"] = soft_label(encoded_examples["probs"] )

        return encoded_examples

    def train(self, set_name=None, eval_set="dev", select_metric="Accuracy", patience=5, soft_labels=False,nla=False):
        set_name = "train" if set_name is None else set_name
        assert select_metric in METRICS2FN

        model_short_name = self.args.model_name_or_path.split('/')[-1]

        cache_folder = os.path.join(self.args.data_path, "cache")
        if not os.path.isdir(cache_folder):
            os.makedirs(cache_folder)
        load_path = os.path.join(cache_folder, f"{model_short_name}-ft-{set_name}-data.pt")

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        if os.path.isfile(load_path) and not self.args.override:
            encoded_train_examples = torch.load(load_path)
        else:
            data_path = os.path.join(self.args.data_path, f"{set_name}.json")
            examples = read_jsonl(data_path)
            self.report_data_stats(examples)
            encoded_train_examples = self.encode_data(tokenizer, examples,nla=nla,soft_labels=soft_labels, with_label=True)
            torch.save(encoded_train_examples, load_path)

        id2label = {id_: label for id_, label in enumerate(self.raw_classes)}
        label2id = {label: id_ for id_, label in id2label.items()}
        config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        config.id2label = id2label
        config.label2id = label2id
        model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name_or_path, **config.__dict__)
        train_dataset = MyDataset(encoded_train_examples)
        if self.device_count == 0:
            device = "cpu"
            self.device_count = 1
        else:
            device = "cuda:0"
            model = DataParallel(model.to(device), device_ids=[i for i in range(self.device_count)])

        train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size_per_device * self.device_count, num_workers=1, shuffle=True)
        total_steps = len(train_loader) * self.args.train_epochs / self.args.accumulation_steps

        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.args.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=self.args.train_training_lr, eps=1e-8)

        # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.pre_train_training_lr, eps=1e-8)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_ratio * total_steps, num_training_steps=total_steps)
        if self.args.train_lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_ratio * total_steps, num_training_steps=total_steps)
        elif self.args.train_lr_scheduler == "linearconstant":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=total_steps)
        else:
            scheduler = get_constant_schedule(optimizer)

        if self.args.multi_label:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        eval_dataset = None
        if eval_set is not None and os.path.isfile(os.path.join(self.args.data_path, f"{eval_set}.json")):
            data_path = os.path.join(self.args.data_path, f"{eval_set}.json")
            examples = read_jsonl(data_path)
            encoded_eval_examples = self.encode_data(tokenizer, examples, nla=nla,with_label=True)
            eval_dataset = MyDataset(encoded_eval_examples)

        model.train().to(device)
        global_step = 0
        eval_loss = 0
        best_score = -1
        patience_check = 0
        save_model_path = os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name, "final_model")
        if nla:
            assert "is_aug" in encoded_train_examples
        self.logger.info(f"************************ Start training ***************************")
        self.logger.info(f"No. of training examples {len(encoded_train_examples['input_ids'])} ({set_name}.json)")
        self.logger.info(f"Batch size {self.args.train_batch_size_per_device * self.device_count}, no. of devices {self.device_count}")
        self.logger.info(f"************************ Start training ***************************")

        for i in range(self.args.train_epochs):
            self.logger.info(f"Epoch {i + 1}:")
            wrap_dataset_loader = tqdm(train_loader)
            model.train()
            model.zero_grad()
            total_epoch_loss = 0
            for j, batch in enumerate(wrap_dataset_loader):
                input_ids = batch["input_ids"].to(device)
                input_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
                logits = outputs.logits
                if not soft_labels:
                    labels = batch["labels"].to(device)
                    loss = loss_fn(logits, labels)
                # soft labels
                else:
                    probs = batch["probs"].to(device)
                    loss = -torch.sum(probs * logits.log_softmax(dim=-1), axis=-1).mean()

                total_epoch_loss += loss.item()
                eval_loss += loss.item()
                loss.backward()
                if (j + 1) % self.args.accumulation_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                global_step += 1
                wrap_dataset_loader.update(1)
                wrap_dataset_loader.set_description(
                    f"Encoder Training - epoch {i + 1}/{self.args.train_epochs} iter {j}/{len(wrap_dataset_loader)}: train loss {loss.item():.8f}. lr {scheduler.get_last_lr()[0]:e}")
                if self.args.train_eval_steps > 0 and global_step % self.args.train_eval_steps == 0:
                    # self.logger.info(f"evaluation during training on {eval_set} set: ")
                    self.logger.info(f"\naverage training loss at global_step={global_step}: {eval_loss / self.args.train_eval_steps}")
                    eval_loss = 0
                    if eval_dataset is not None:
                        self.inference(model, eval_dataset)
                        model.train()

            self.logger.info(f"Average training loss for epoch {i + 1}: {total_epoch_loss / len(train_loader)}")

            if nla:
                # get aug examples from encoded_train_examples and keep a record of their indexes
                # load them into MyDataset -> DataLoader
                # use the model at this checkpoint to get the predictions
                # get the indexes of those whose predictions are considered to be noisy, prob anneals from 0.9 to 1/k (k equals the number of classes, the annealing is based on epoch number here linearly)
                # remove them from encoded_train_examples
                # load the new encoded_train_examples into MyDataset -> DataLoader
                # train_loader is used in next epoch!!!!
                '''
                Ref Appendix A: https://arxiv.org/pdf/2109.09193.pdf
                '''
                self.logger.info("**************** start noise label annealing")
                if isinstance(encoded_train_examples["is_aug"], list):
                    encoded_train_examples["is_aug"] = torch.tensor(encoded_train_examples["is_aug"])
                if isinstance(encoded_train_examples["labels"], list):
                    encoded_train_examples["labels"] = torch.tensor(encoded_train_examples["labels"])

                aug_indexes = (encoded_train_examples["is_aug"] == 1).nonzero().squeeze()
                aug_mask = encoded_train_examples["is_aug"].bool()
                aug_labels = encoded_train_examples["labels"][aug_mask]
                aug_input_ids = encoded_train_examples["input_ids"][aug_mask]
                aug_attention_masks = encoded_train_examples["attention_mask"][aug_mask]
                model.eval()
                to_remove_indexes = []
                p = 0.9 - (0.9 - 1 / len(self.raw_classes)) * i / self.args.train_epochs
                with torch.no_grad():
                    for i_, index in enumerate(aug_indexes):
                        aug_label = aug_labels[i_]
                        aug_input_id = aug_input_ids[i_].unsqueeze(0).to(device)
                        aug_attention_mask = aug_attention_masks[i_].unsqueeze(0).to(device)
                        outputs = model(aug_input_id, attention_mask=aug_attention_mask, return_dict=True)
                        logits = outputs.logits
                        probs = torch.nn.Softmax(dim=-1)(logits)
                        pred_indice = probs.argmax(-1).tolist()[0]
                        prob = probs[0][pred_indice].item()
                        if prob > p and pred_indice != aug_label.item():
                            to_remove_indexes.append(index.item())
                model.train()
                if len(to_remove_indexes) > 0:
                    to_remove_indexes = torch.tensor(to_remove_indexes)
                    remove_mask = torch.tensor([1] * len(encoded_train_examples["is_aug"]))
                    remove_mask.index_fill_(0, to_remove_indexes, 0)
                    remove_mask = remove_mask.bool()
                    new_encoded_train_examples = {}
                    for k, v in encoded_train_examples.items():
                        new_encoded_train_examples[k] = v[remove_mask]
                    encoded_train_examples = new_encoded_train_examples
                    train_dataset = MyDataset(encoded_train_examples)
                    train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size_per_device * self.device_count, num_workers=1, shuffle=True)
                self.logger.info("**************** end noise label annealing")

            # evaluate at the end of epoch if eval_steps is smaller than or equal to 0
            # if self.args.train_eval_steps <= 0:
            #     self.logger.info(f"evaluation during training on {eval_set} set: ")
            # if eval_dataset is not None:
            #     self.inference(model, eval_dataset)
            #     model.train()
            if self.args.save_last_n_epochs > 0:
                if self.args.train_epochs > self.args.save_last_n_epochs:
                    if i + 1 > self.args.train_epochs - self.args.save_last_n_epochs:
                        # save up only the last n epochs if train_epochs>=self.args.save_last_n_epochs
                        self.save_ck(model, tokenizer, model_short_name, i, eval_dataset)
                else:
                    # save up at end of each epoch if train_epochs <=self.args.save_last_n_epochs
                    self.save_ck(model, tokenizer, model_short_name, i, eval_dataset)
            else:
                self.logger.info(f"evaluation during training on {eval_set} set ({model_short_name}_epoch{i + 1}): ")
                eval_scores = self.inference(model, eval_dataset)
                current_select_score = eval_scores[select_metric]
                if current_select_score > best_score:
                    self.logger.info(f"******** found best checkpoint at epoch_{i + 1} based on {select_metric}={current_select_score} versus the best = {best_score} on eval set, so save the checkpoint to: {save_model_path} ******** ")
                    if isinstance(model, DataParallel):
                        model.module.save_pretrained(save_model_path)
                    else:
                        model.save_pretrained(save_model_path)
                    tokenizer.save_pretrained(save_model_path)
                    best_score = current_select_score
                    patience_check = 0
                else:
                    self.logger.info(f"******** not the best checkpoint based on {select_metric}={current_select_score} versus the best = {best_score} on eval set, so increase patience index by 1 ******** ")
                    patience_check += 1

                self.logger.info(f"******** Patience: {patience_check}/{patience} ******** ")
                if patience_check >= patience:
                    self.logger.info(f"Patience: {patience_check}/{patience}: run out of patience, so training stopped early at epoch_{i + 1}: ")
                    break
        # save up at end of training!
        # trained_model_path = os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name, "final_model")
        # if isinstance(model, DataParallel):
        #     model.module.save_pretrained(trained_model_path)
        # else:
        #     model.save_pretrained(trained_model_path)
        # tokenizer.save_pretrained(trained_model_path)
        return save_model_path
        # self.logger.info(f"evaluation on eval set with trained model: {save_model_path}")
        # if eval_set is not None:
        #     return self.predict(load_path=save_model_path, set_name=eval_set)
        # else:
        #     return {}

    def save_ck(self, model, tokenizer, model_short_name, i, eval_dataset, save_at="epoch"):
        if isinstance(model, DataParallel):
            model.module.save_pretrained(os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name, f"{save_at}_{i + 1}"))
        else:
            model.save_pretrained(os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name, f"{save_at}_{i + 1}"))
        tokenizer.save_pretrained(os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name, f"{save_at}_{i + 1}"))
        dev_eval = self.inference(model, eval_dataset)
        dev_eval.update({"save_at": f"{save_at}_{i + 1}"})
        json.dump(dev_eval, open(os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name, f"{save_at}_{i + 1}", "dev_eval.json"), "w"))

    def select_best_ck(self, select_metric="binary-F1"):
        model_short_name = self.args.model_name_or_path.split('/')[-1]
        model_save_folder = os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name)
        ck_names = os.listdir(os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name))
        best_ck_name = "final_model"
        score = -100000
        for ck_name in ck_names:
            if ck_name != "final_model":
                dev_eval = json.load(open(os.path.join(model_save_folder, ck_name, "dev_eval.json")))
                if dev_eval[select_metric] >= score:
                    score = dev_eval[select_metric]
                    best_ck_name = ck_name
        return os.path.join(model_save_folder, best_ck_name)

    def predict(self, load_path=None, set_name="test", pred2file=False, with_label=True, save_tag="", label_surface_name=False):
        # load up
        # model_short_name = self.data_args.model_for_preselftrain.split('/')[-1]
        # model_path = os.path.join(self.training_args.output_path, "self_train", model_short_name, f"epoch_1")
        # load_path = model_path if load_path is None else load_path
        if save_tag == "":
            save_tag = f"seed_{self.args.seed}"
        model_short_name = self.args.model_name_or_path.split('/')[-1]
        if load_path is None:
            load_path = os.path.join(self.args.output_path, self.__class__.__name__, str(self.args.seed), model_short_name, "final_model")
        ck_name = load_path.split(os.path.sep)[-1]
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        data_path = os.path.join(self.args.data_path, f"{set_name}.json")
        examples = read_jsonl(data_path)
        encoded_eval_examples = self.encode_data(tokenizer, examples, with_label=with_label)
        eval_dataset = MyDataset(encoded_eval_examples)
        if pred2file:
            scores_dict, preds = self.inference(model, eval_dataset, return_preds=True)
            preds_folder = os.path.join(self.args.output_path, "preds", model_short_name)
            if not os.path.isdir(preds_folder):
                os.makedirs(preds_folder)

            # for error analysis purpose when the labels are available
            if with_label:
                predfile = os.path.join(preds_folder, f"{ck_name}_{set_name}_pred_{save_tag}.csv")

            predfile_nogt = os.path.join(preds_folder, f"{ck_name}_{set_name}_pred_nogt_{save_tag}.csv")
            text_field = "text" if "text" in examples[0] else "content"
            df = pd.DataFrame(columns=[text_field, "label"])
            df[text_field] = [example[text_field] for example in examples]
            if with_label:
                df["label"] = [example["label"] for example in examples]
            else:
                df["label"] = ["-1" for _ in range(len(examples))]

            if "language" in examples[0]:
                df["language"] = [example["language"] for example in examples]

            if label_surface_name:
                df["prediction"] = [self.raw_classes[pred] for pred in preds]
            else:
                df["prediction"] = preds

            if with_label:
                df.to_csv(predfile, index=False)

            del df["label"]

            if "language" in examples[0]:
                del df["language"]

            df.to_csv(predfile_nogt, index=False)
            self.logger.info(f"predictions are written to: {predfile_nogt}")
        else:
            scores_dict = self.inference(model, eval_dataset)
        return scores_dict

    def inference(self, model, dataset, return_preds=False):
        start_infer_time = time.time()
        eval_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=1, shuffle=False)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            model.eval().to(device)
            gts, preds = [], []
            with torch.no_grad():
                wrap_loader = tqdm(eval_loader, desc="predicting")
                for j, batch in enumerate(wrap_loader):
                    input_ids = batch["input_ids"].to(device)
                    input_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
                    logits = outputs.logits
                    # IF MULTI-LABEL
                    if self.args.multi_label:
                        probs = logits.sigmoid()
                        preds.extend((probs > 0.5).int().tolist())
                        if "labels" in batch:
                            # one hot
                            gts.extend(batch["label_indices"].int().tolist())
                    else:
                        probs = torch.nn.Softmax(dim=-1)(logits)
                        pred_indices = probs.argmax(-1).tolist()
                        preds.extend([self.raw_classes[i] for i in pred_indices])
                        if "labels" in batch:
                            # label indices
                            gts.extend([self.raw_classes[i] for i in batch["labels"]])
                    # wrap_loader.update(1)
                    # wrap_loader.set_description("predicting")
                self.logger.info(f"inference time (s): {time.time() - start_infer_time}")
                if len(gts) == len(preds):
                    self.logger.info(classification_report(gts, preds, digits=4))
                    self.logger.info(f"accuracy score: {accuracy_score(gts, preds)}")
                    if not self.args.multi_label and len(set(gts)) == 2:
                        preds = [self.raw_classes.index(i) for i in preds]
                        gts = [self.raw_classes.index(i) for i in gts]
                    return_dict = calculate_perf(preds, gts)
                    self.logger.info(json.dumps(return_dict, indent=2))
                    if return_preds:
                        return return_dict, preds
                    return return_dict
                if return_preds:
                    return {}, [self.raw_classes.index(i) for i in preds]
                return {}, []
        except RuntimeError as err:
            self.logger.info(f"GPU memory is not enough: {err}")

    def report_data_stats(self, examples):
        tmp_list = [len(example["text"].split(" ") if "text" in example else example["content"].split(" ")) for example in examples]
        max_ex_len = max(tmp_list)
        avg_ex_len = np.average(tmp_list)
        self.logger.info("#############Data stats#####################")
        self.logger.info('Example max length: {} (words)'.format(max_ex_len))
        self.logger.info('Example average length: {} (words)'.format(avg_ex_len))
        exceed_count = len([i for i in tmp_list if i > self.args.max_seq_length])
        self.logger.info(f'Examples with words beyond max_seq_length ({self.args.max_seq_length}): {exceed_count}/{len(examples)} (examples)')
        self.logger.info("##################################")
