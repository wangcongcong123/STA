from src.t2t import T2TArgs, T2TTrainer


def get_trainer(do_train=True,
                gen_input_rep_k=3,
                train_training_lr=5e-5,
                optimizer="adamw",
                do_generate=False,
                seed=2021,
                train_batch_size_per_device=16,
                train_epochs=32,
                prompt_topic="sentiment",
                train_set_name="train_10",
                datapath=None,
                two_prompts=False,
                model_path=None):
    if two_prompts:
        gen_input_rep_k = 0

    args = T2TArgs(
        # data args
        data_path=datapath,
        two_prompts=two_prompts,
        output_path=datapath,
        max_seq_length=128,
        max_tgt_seq_length=64,
        train_batch_size_per_device=train_batch_size_per_device,
        train_epochs=train_epochs,
        save_at_epoch=False,
        encoding_batch_size=1000,
        # training args
        model_name_or_path=model_path,
        accumulation_steps=1,
        train_training_lr=train_training_lr,
        train_lr_scheduler="linear",  # linear,linearconstant, constant
        optimizer=optimizer,  # adamw, adafactor
        override=True,
        seed=seed,
        gen_input_rep_k=gen_input_rep_k,
        eval_batch_size=128,
        warmup_ratio=0.1,
        train_eval_steps=-1,
    )
    trainer = T2TTrainer(args)
    if do_train:
        trainer.train(set_name=train_set_name, prompt_topic=prompt_topic)
    if do_generate:
        trainer.seq2seq_aug(f"{datapath}/{train_set_name}.json", f"{datapath}/{train_set_name}_aug3.json")
    return trainer
