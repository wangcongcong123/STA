import json, os

DEVICES = "0"  # or you can use mutiple gpus here: 0,1,2,3
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES
from run_train import train_pipeline

if __name__ == '__main__':
    # configuration
    relative_path = "data/"
    dataset_name = "humaid"
    self_control = True
    twoprompts = False
    config = {
        "devices": DEVICES,
        "relative_path": relative_path,
        "dataset_name": dataset_name,
        "downsample_ratios": [5, 10, 20, 50, 100],
        "aug_extents": [1, 2, 3, 4, 5],
        "ds_model_paths": ["bert-base-uncased"],
        "datapath": f"{relative_path}/{dataset_name}",
        "prompt_topic": "disaster aid types",
        "select_metric": "Accuracy",
        "gen_model_path": "t5-base"
    }
    random_seeds = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
    ex_results = {}
    gen_res = {}
    for random_seed in random_seeds:
        config.update({"random_seed": random_seed})
        results = train_pipeline(config, self_control=self_control, two_prompts=twoprompts)
        gen_res.update({f"random_seed:{random_seed}": results})
        print(f"**** results of random seeds:******")
        print(json.dumps(gen_res, indent=2))

    ex_results.update({f"gen_model_path:{os.path.basename(config['gen_model_path'])}": gen_res})
    print(f"**** experiments for augmentation (down-sampling control):******")
    print(json.dumps(ex_results, indent=2))

# nohup python train_emotion.py > train_emotion_self_2021-2030.out &
