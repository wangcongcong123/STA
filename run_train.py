from src.encoder import EncoderArgs, EncoderTrainer
import os, json, time
from train_t2t import get_trainer as get_t2t_trainer
from src.utils import split_out

def train_pipeline(config, do_gen_train=True, ds_train_epochs=20, self_control=True, two_prompts=False):
    ex_results = {}
    print(f"config: {json.dumps(config, indent=2)}")
    ex_results.update({"config": config})
    if do_gen_train:
        do_train = config['aug_extents'] != [0]
    else:
        do_train = do_gen_train

    for downsample_ratio in config['downsample_ratios']:
        print("down-sampling the original training set")
        fp, downsample_fn = split_out(ratio=downsample_ratio, filepath=f"{config['datapath']}/train.json", random_seed=config['random_seed'])
        train_epochs = 32
        augmentor = get_t2t_trainer(
            do_train=do_train,
            two_prompts=two_prompts,
            gen_input_rep_k=3,
            train_epochs=train_epochs,
            seed=config['random_seed'],
            prompt_topic=config['prompt_topic'],
            train_batch_size_per_device=int(16 / len(config['devices'].split(","))) + 1,  # small training size works better for small datasets
            do_generate=False,
            train_set_name=downsample_fn,
            datapath=config['datapath'],
            model_path=config["gen_model_path"]
        )
        for aug_extent in config['aug_extents']:
            if aug_extent == 0:
                aug_method = "no_aug"
                train_set_name = downsample_fn
            else:
                aug_method = f"aug_mc_{aug_extent}"
                train_set_name = downsample_fn + f"_{aug_extent}_aug"
                if config["self_control"]:
                    train_set_name += f"_self_{config['random_seed']}"
                else:
                    train_set_name += f"_noself_{config['random_seed']}"
                if two_prompts:
                    train_set_name += "_twoprompts"

                augmentor.seq2seq_aug(fp, f"{config['datapath']}/{train_set_name}.json", prompt_topic=config['prompt_topic'], self_control=self_control, aug_extent=aug_extent)

            for model_path in config['ds_model_paths']:
                args = EncoderArgs(
                    # data args
                    data_path=config['datapath'],
                    output_path=config['datapath'],
                    max_seq_length=128,
                    train_batch_size_per_device=int(16 / len(config['devices'].split(","))) + 1,
                    train_epochs=ds_train_epochs,
                    encoding_batch_size=1000,
                    # training args
                    model_name_or_path=model_path,
                    accumulation_steps=1,
                    train_training_lr=2e-5,
                    train_lr_scheduler="linear",  # linear, constant
                    eval_batch_size=128,
                    warmup_ratio=0.1,
                    override=True,
                    seed=config['random_seed'],
                    train_eval_steps=-1,
                )
                trainer = EncoderTrainer(args)
                test_set_name = "test"
                pred_results = trainer.predict(set_name=test_set_name, pred2file=True, with_label=True, save_tag=f"seed_{config['random_seed']}")
                ex_results.update({f"{config['dataset_name']}-{train_set_name}-{aug_method}-downsample_ratio({downsample_ratio})-{test_set_name}-seed-{config['random_seed']}-{os.path.basename(model_path)}": pred_results})
                print("results-report-in:" + json.dumps(ex_results, indent=2))

    print(f"* experiments using t5-base for augmentation *")
    print(json.dumps(ex_results, indent=2))
    return ex_results
