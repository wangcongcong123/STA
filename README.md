## The code for the paper ["STA: Self-controlled Text Augmentation for Improving Text Classification"](http://arxiv.org/abs/2302.12784)

To run this code, make sure you have

```shell
pip install transformers==4.10.0
```

The datasets in the work include `SST-2`, `EMOTION`, `TREC` and `HumAID`. They should be put in `./data/` (they are also downloadable from [https://huggingface.co/datasets](https://huggingface.co/datasets). For `HumAID`, the dataset can be downloaded from: [https://crisisnlp.qcri.org/humaid_dataset](https://crisisnlp.qcri.org/humaid_dataset)).

To run STA on each dataset, simply execute the following commands

```shell
nohup python train_sst2.py > train_sst2_self_2021-2030.out &
nohup python train_emotion.py > train_emotion_self_2021-2030.out &
nohup python train_trec.py > train_trec_self_2021-2030.out &
nohup python train_humaid.py > train_humaid_self_2021-2030.out &
```

To run other variants such as `STA-noself`, just change the variable `self_control` from `True` to `False` in the corresponding scripts.
* It is expected to obtain better results using larger generation and downstream models such as `roberta-large` and `t5-large` in the scripts


### To cite this paper
```
@article{wang2023sta,
  title={STA: Self-controlled Text Augmentation for Improving Text Classification},
  author={Congcong Wang, Gonzalo Fiz Pontiveros, Steven Derby and Tri Kurniawan Wijaya},
  journal={arXiv preprint arXiv:2302.12784},
  year={2023}
}
```
