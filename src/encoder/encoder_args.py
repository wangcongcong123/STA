class EncoderArgs:
    '''
    a Args class that maintain all arguments for data, model training and inference
    '''
    # data args
    data_path = ""
    max_seq_length = 128
    encoding_batch_size = 1000
    multi_label = False

    # training args
    output_path = ""
    model_name_or_path = ""
    train_batch_size_per_device = 16

    train_epochs = 3
    # for storage concern
    save_last_n_epochs = -1

    accumulation_steps = 1
    train_training_lr = 2e-5

    warmup_ratio = 0.1
    eval_batch_size = 32
    train_lr_scheduler = "linear"

    seed = 2021
    weight_decay = 0.1

    override = False
    save_at_epoch = True
    train_eval_steps = -1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
