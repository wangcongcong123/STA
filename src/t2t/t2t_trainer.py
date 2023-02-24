from transformers import get_constant_schedule, get_constant_schedule_with_warmup, AdamW, Adafactor, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from src.t2t import T2TArgs
from src.datasets import MyDataset
from copy import deepcopy
from src.utils import *
import json, sys
import transformers
from torch.utils.data import DataLoader
import pandas as pd
import time
from torch.nn.parallel import DataParallel

KEY2MODELLOADER = {
    "t5-small": T5ForConditionalGeneration,
    "t5-base": T5ForConditionalGeneration,
    "t5-large": T5ForConditionalGeneration,

}

KEY2TOKENIZER = {
    "t5-small": T5Tokenizer,
    "t5-base": T5Tokenizer,
    "t5-large": T5Tokenizer,
}

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class T2TTrainer():
    def __init__(self, args: T2TArgs):
        # transformers.logging.set_verbosity_info()
        self.logger = transformers.logging.get_logger()
        self.logger.setLevel(logging.INFO)
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

    def encode_data(self, tokenizer, examples, with_label=False):
        self.report_data_stats(examples, tokenizer=tokenizer)
        encoded_examples = {}
        text_field = "text" if "text" in examples[0] else "content"
        for i in trange(0, len(examples), self.args.encoding_batch_size, desc="encoding data...."):
            batch_examples = examples[i:i + self.args.encoding_batch_size]
            input_texts = [be[text_field] for be in batch_examples]
            inputs = tokenizer(input_texts, padding="max_length", return_tensors="pt", truncation=True, max_length=self.args.max_seq_length)
            if with_label:
                label_texts = [be["label"] for be in batch_examples]
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(label_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=self.args.max_tgt_seq_length)
                inputs.update({"labels": labels["input_ids"]})

            for k, v in inputs.items():
                if k not in encoded_examples:
                    encoded_examples[k] = v
                else:
                    encoded_examples[k] = torch.cat([encoded_examples[k], v], dim=0)
        return encoded_examples

    def generate_examples_by_templates(self, example, topic="sentiment", label2examples=None):
        new_examples = []
        new_examples.append({"text": f"Given {topic}: {', '.join(self.raw_classes)}. Classify: {example['text']}", "label": example["label"]})
        if not self.args.two_prompts:
            new_examples.append({"text": f"Text: {example['text']} Is this text about {example['label']} {topic}?", "label": "yes"})
            classes = deepcopy(self.raw_classes)
            classes.remove(example["label"])
            new_examples.append({"text": f"Text: {example['text']} Is this text about {random.sample(classes, 1)[0]} {topic}?", "label": "no"})
            if label2examples is None:
                return new_examples

        new_examples.append({"text": f"Description: {example['label']} {topic}. Text:", "label": example['text']})

        if not self.args.two_prompts:
            num = self.args.gen_input_rep_k
            candidates = deepcopy(label2examples[example["label"]])
            candidates.remove(example)
            assert len(candidates) >= num
            for _ in range(num):
                another_example = random.sample(candidates, 1)[0]
                words = another_example['text'].split()
                k = 3
                if len(words) < 3:
                    k = len(words)
                t1, t2 = " ".join(words[:k]), " ".join(words[k:])
                new_examples.append({"text": f"Description: {example['label']} {topic}. Text: {example['text']} Another text: {t1}", "label": t2})
                candidates.remove(another_example)
        return new_examples

    def encode_examples_with_multiple_prompts(self, tokenizer, examples, prompt_topic="sentiment"):
        label2examples = {}
        for example in examples:
            if example["label"] not in label2examples:
                label2examples[example["label"]] = []
            label2examples[example["label"]].append(example)

        formed_examples = []
        for example in examples:
            formed_examples.extend(self.generate_examples_by_templates(example, topic=prompt_topic, label2examples=label2examples))
        return self.encode_data(tokenizer, formed_examples, with_label=True)

    def train(self, set_name=None, eval_set=None, prompt_topic="sentiment"):
        set_name = "train" if set_name is None else set_name
        model_short_name = self.args.model_name_or_path.split('/')[-1]
        load_path = os.path.join(self.args.data_path, f"{model_short_name}-ft-{set_name}-data.pt")
        tokenizer = KEY2TOKENIZER[model_short_name].from_pretrained(self.args.model_name_or_path)
        if os.path.isfile(load_path) and not self.args.override:
            encoded_train_examples = torch.load(load_path)
        else:
            data_path = os.path.join(self.args.data_path, f"{set_name}.json")
            examples = read_jsonl(data_path)
            encoded_train_examples = self.encode_data(tokenizer, examples, with_label=True)
            torch.save(encoded_train_examples, load_path)

        model = KEY2MODELLOADER[model_short_name].from_pretrained(self.args.model_name_or_path)
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
        if self.args.optimizer == "adamw":
            optimizer = AdamW(optim_groups, lr=self.args.train_training_lr, eps=1e-8)
        else:
            optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.args.train_training_lr)

        if self.args.train_lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_ratio * total_steps, num_training_steps=total_steps)
        elif self.args.train_lr_scheduler == "linearconstant":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_ratio * total_steps)
        else:
            scheduler = get_constant_schedule(optimizer)

        eval_dataset = None
        if eval_set is not None and os.path.isfile(os.path.join(self.args.data_path, f"{eval_set}.json")):
            data_path = os.path.join(self.args.data_path, f"{eval_set}.json")
            examples = read_jsonl(data_path)
            encoded_eval_examples = self.encode_data(tokenizer, examples, with_label=True)
            eval_dataset = MyDataset(encoded_eval_examples)

        model.train().to(device)
        global_step = 0
        eval_loss = 0
        self.logger.info(f"No. of training examples {len(encoded_train_examples['input_ids'])}")
        self.logger.info(f"Batch size {self.args.train_batch_size_per_device * self.device_count}, no. of devices {self.device_count}")
        display_k = 10
        self.logger.info(f"********* display {display_k} training examples for sanity checking *********")
        for src, tgt in zip(encoded_train_examples["input_ids"][:display_k], encoded_train_examples["labels"][:display_k]):
            self.logger.info("-----------------------------------------------------------")
            self.logger.info(f"src: {tokenizer.decode(src, skip_special_tokens=True)}")
            self.logger.info(f"tgt: {tokenizer.decode(tgt, skip_special_tokens=True)}")

        for i in range(self.args.train_epochs):
            self.logger.info(f"Epoch {i + 1}:")
            wrap_dataset_loader = tqdm(train_loader)
            model.zero_grad()
            total_epoch_loss = 0
            for j, batch in enumerate(wrap_dataset_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss.mean()
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
                    f"T2T Training - epoch {i + 1}/{self.args.train_epochs} iter {j}/{len(wrap_dataset_loader)}: train loss {loss.item():.8f}. lr {scheduler.get_last_lr()[0]:e}")
                if self.args.train_eval_steps > 0 and global_step % self.args.train_eval_steps == 0:
                    self.logger.info(f"\naverage training loss at global_step={global_step}: {eval_loss / self.args.train_eval_steps}")
                    eval_loss = 0
                    if eval_dataset is not None:
                        self.inference(model, tokenizer, eval_dataset)
                        model.train()
            train_loss = total_epoch_loss / len(train_loader)
            self.logger.info(f"Average training loss for epoch {i + 1}: {train_loss}")

            if self.args.train_eval_steps <= 0:
                self.logger.info(f"evaluation during training on {eval_set} set: ")
                if eval_dataset is not None:
                    self.inference(model, tokenizer, eval_dataset)
                    model.train()

            if self.args.train_epochs > 3:
                if i + 1 > self.args.train_epochs - 3:
                    # save up only the last three epochs if train_epochs>=3
                    self.save_ck(model, tokenizer, model_short_name, i, eval_dataset, train_loss)
            else:
                # save up at end of each epoch if train_epochs <=3
                self.save_ck(model, tokenizer, model_short_name, i, eval_dataset, train_loss)

        # save up at end of training!
        trained_model_path = os.path.join(self.args.output_path, self.__class__.__name__, model_short_name, "final_model")
        if isinstance(model, DataParallel):
            model.module.save_pretrained(trained_model_path)
        else:
            model.save_pretrained(trained_model_path)

        tokenizer.save_pretrained(trained_model_path)
        self.logger.info(f"evaluation on test set with trained model at last epoch: {trained_model_path}")
        if eval_set is not None:
            return self.predict(load_path=trained_model_path, set_name=eval_set)
        else:
            return {}

    def save_ck(self, model, tokenizer, model_short_name, i, eval_dataset, train_loss, save_at="epoch"):
        if isinstance(model, DataParallel):
            model.module.save_pretrained(os.path.join(self.args.output_path, self.__class__.__name__, model_short_name, f"{save_at}_{i + 1}"))
        else:
            model.save_pretrained(os.path.join(self.args.output_path, self.__class__.__name__, model_short_name, f"{save_at}_{i + 1}"))
        tokenizer.save_pretrained(os.path.join(self.args.output_path, self.__class__.__name__, model_short_name, f"{save_at}_{i + 1}"))
        dev_eval = {"train_loss_inv": -train_loss}  # that is the opposite of train_loss since we want to keep all scores greater is better when selecting models
        if eval_dataset is not None:
            dev_eval = self.inference(model, tokenizer, eval_dataset)

        dev_eval.update({"save_at": f"{save_at}_{i + 1}"})
        json.dump(dev_eval, open(os.path.join(self.args.output_path, self.__class__.__name__, model_short_name, f"{save_at}_{i + 1}", "dev_eval.json"), "w"))

    def predict(self, load_path=None, set_name="test", pred2file=False, with_label=True, save_tag="epoch", label_surface_name=False):
        # load up
        # model_short_name = self.args.model_for_preselftrain.split('/')[-1]
        # model_path = os.path.join(self.args.output_path, "self_train", model_short_name, f"epoch_1")
        # load_path = model_path if load_path is None else load_path
        model_short_name = self.args.model_name_or_path.split('/')[-1]
        if load_path is None:
            load_path = os.path.join(self.args.output_path, self.__class__.__name__, model_short_name, "final_model")
        ck_name = load_path.split(os.path.sep)[-1]
        tokenizer = T5Tokenizer.from_pretrained(load_path)
        model = KEY2MODELLOADER[model_short_name].from_pretrained(load_path)
        data_path = os.path.join(self.args.data_path, f"{set_name}.json")
        examples = read_jsonl(data_path)
        encoded_eval_examples = self.encode_data(tokenizer, examples, with_label=with_label)
        eval_dataset = MyDataset(encoded_eval_examples)
        if pred2file:
            scores_dict, preds = self.inference(model, tokenizer, eval_dataset, return_preds=True)
            preds_folder = os.path.join(self.args.output_path, "preds", model_short_name)
            if not os.path.isdir(preds_folder):
                os.makedirs(preds_folder)

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
            df.to_csv(predfile, index=False)

            del df["label"]

            if "language" in examples[0]:
                del df["language"]

            df.to_csv(predfile_nogt, index=False)
            self.logger.info(f"predictions are written to: {predfile}")
        else:
            scores_dict = self.inference(model, tokenizer, eval_dataset)
        return scores_dict

    def inference(self, model, tokenizer, dataset, return_preds=False):
        start_infer_time = time.time()
        eval_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=1, shuffle=False)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            model.eval().to(device)
            gts, preds = [], []
            with torch.no_grad():
                wrap_loader = tqdm(eval_loader, desc="predicting")
                for j, batch in enumerate(wrap_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if "labels" in batch:
                        labels = batch.pop("labels")
                        gts.extend([tokenizer.decode(each, skip_special_tokens=True, max_length=self.args.max_tgt_seq_length) for each in labels])

                    if isinstance(model, DataParallel):
                        outputs = model.module.generate(**batch, max_length=self.args.max_tgt_seq_length + 2)
                    else:
                        outputs = model.generate(**batch, max_length=self.args.max_tgt_seq_length + 2)
                    # results = tokenizer.decode(outputs["input_ids"])
                    for each in outputs:
                        pred = tokenizer.decode(each, skip_special_tokens=True, max_length=self.args.max_tgt_seq_length)
                        pred = self.raw_classes[0] if pred not in self.raw_classes else pred
                        preds.append(pred)
                    # print(len(results))
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

    def report_data_stats(self, examples, tokenizer=None):
        self.logger.info("calculating data stats...")
        if tokenizer is not None:
            tmp_list = [len(list(tokenizer(example["text"] if "text" in example else example["content"])["input_ids"])) for example in tqdm(examples, desc="calculating data stats...")]
        else:
            tmp_list = [len(example["text"].split(" ") if "text" in example else example["content"].split(" ")) for example in examples]
        max_ex_len = max(tmp_list)
        avg_ex_len = np.average(tmp_list)
        self.logger.info("#############Data stats on source sequence #####################")
        self.logger.info('Example max length: {} (words)'.format(max_ex_len))
        self.logger.info('Example average length: {} (words)'.format(avg_ex_len))
        exceed_count = len([i for i in tmp_list if i > self.args.max_seq_length])
        self.logger.info(f'Examples with words beyond max_seq_length ({self.args.max_seq_length}): {exceed_count}/{len(examples)} (examples)')
        self.logger.info("##################################")

    def seq2seq_aug(self, src_file, to_file, aug_extent=1, max_length=128, prompt_topic="sentiment", model_load_path=None, self_control=True, dedup=True, temperature=1.0):
        model_short_name = self.args.model_name_or_path.split('/')[-1]
        if model_load_path is None:
            model_load_path = os.path.join(self.args.output_path, self.__class__.__name__, model_short_name, "final_model")
        tokenizer = KEY2TOKENIZER[model_short_name].from_pretrained(model_load_path)
        model = KEY2MODELLOADER[model_short_name].from_pretrained(model_load_path)
        exs = read_jsonl(src_file)

        label2examples = {}
        for ex in exs:
            one_hot = len(self.raw_classes) * [0.0]
            one_hot[self.raw_classes.index(ex["label"])] = 1.0
            ex["probs"] = one_hot
            ex["classes"] = self.raw_classes
            ex["aug"] = False
            if ex["label"] not in label2examples:
                label2examples[ex["label"]] = []
            label2examples[ex["label"]].append(ex)

        label2otherexamples = {}
        for label in label2examples:
            other_examples = [each for each in exs if each["label"] != label]
            label2otherexamples[label] = other_examples

        all_augs = []
        all_examples = []
        model.cuda()
        global_step = 0
        early_stop = 20
        confidence_select_magnifier = 5
        total_step = int(sum(len(examples) * aug_extent for examples in label2examples.values()))
        total_step = total_step * confidence_select_magnifier

        print(f"aug for {src_file}")
        if aug_extent > 0:
            for label_name, examples in label2examples.items():
                target_num = int(len(examples) * aug_extent) * confidence_select_magnifier
                augs = []
                last_augs_num = -1
                no_more_count = 0
                exist_texts = [each["text"] for each in examples]

                while len(augs) < target_num and no_more_count < early_stop:
                    # if greedy_decoding:
                    if len(augs) == last_augs_num:
                        # no more augmented examples in 20 consecutive times for greedy decoding generation, then stop generating
                        print(f"no more increase in {no_more_count + 1} consecutive times when decoding: {global_step}/{total_step}")
                        no_more_count += 1
                    else:
                        no_more_count = 0
                    last_augs_num = len(augs)

                    selected_example = random.sample(examples, 1)[0]
                    formed_examples = self.generate_examples_by_templates(selected_example, topic=prompt_topic, label2examples=label2examples)
                    prompts = [each["text"] for each in formed_examples]
                    if self.args.gen_input_rep_k == 0:
                        prompts_aug = prompts[-1:]
                    else:
                        prompts_aug = prompts[-self.args.gen_input_rep_k - 1:-self.args.gen_input_rep_k]

                    input_ids = tokenizer(prompts_aug, return_tensors="pt", padding=True).input_ids.cuda()
                    if input_ids.shape[1] + 8 > max_length:
                        continue
                    else:
                        max_length = min(max_length, input_ids.shape[1] + 64)
                    num_return_sequences = aug_extent * confidence_select_magnifier
                    gen_ex_texts = tokenizer.batch_decode(model.generate(input_ids, do_sample=True, top_k=40, temperature=temperature, max_length=max_length, num_return_sequences=num_return_sequences), skip_special_tokens=True)

                    for gen_ex_text in gen_ex_texts:
                        def dedup_self_select(global_step):
                            this_formed_examples = self.generate_examples_by_templates({"text": gen_ex_text, "label": label_name}, topic=prompt_topic, label2examples=None)
                            input_texts = [each["text"] for each in this_formed_examples[:1]]
                            input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).input_ids.cuda()
                            gen_outs = model.generate(input_ids, max_length=10, output_scores=True, return_dict_in_generate=True)
                            classes_logits = []
                            for ot in self.raw_classes:
                                classes_logits.append(gen_outs.scores[0][0][tokenizer(ot)["input_ids"][0]].item())
                            probs = torch.tensor(classes_logits).softmax(0)
                            probs = probs.tolist()
                            confidence = probs[self.raw_classes.index(label_name)]
                            augs.append({"text": gen_ex_text, "label": label_name, "aug": True, "probs": probs, "confidence": confidence, "classes": self.raw_classes})
                            global_step += 1
                            exist_texts.append(gen_ex_text)
                            if global_step % 10 == 0:
                                print(f"progress: {global_step}/{total_step} (magnifier={confidence_select_magnifier})")

                        if dedup:
                            if len(gen_ex_text) > 16 and gen_ex_text not in exist_texts:
                                global_step += 1
                                dedup_self_select(global_step)
                        else:
                            global_step += 1
                            dedup_self_select(global_step)
                if self_control:
                    sorted_augs_by_confidence = sorted(augs, key=lambda k: k["confidence"], reverse=True)
                    augs = sorted_augs_by_confidence[:int(len(examples) * aug_extent)]
                else:
                    augs = augs[:int(len(examples) * aug_extent)]
                self.logger.info(f'original no. of examples of class **{label_name}**: {len(examples)}')
                self.logger.info(f'augmented no. of examples of class **{label_name}**: {len(augs)}')
                all_augs.extend(augs)
            all_examples.extend(exs + all_augs)
        else:
            all_examples = exs

        with open(to_file, "w+") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")
        self.logger.info(f'augmented data written to: {to_file}')
