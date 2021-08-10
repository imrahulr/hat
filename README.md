# Helper-based Adversarial Training

This repository contains the code for the paper "[Helper-based Adversarial Training: Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off](https://openreview.net/forum?id=BuD2LmNaU3a)" by Rahul Rade and Seyed-Mohsen Moosavi-Dezfooli. 

A short version of the paper has been accepted at the [ICML 2021 Workshop on A Blessing in Disguise: The Prospects and Perils of Adversarial Machine Learning](https://advml-workshop.github.io/icml2021/).

## Setup

### Requirements

Our code has been implemented and tested with `Python 3.8.5` and `PyTorch 1.8.0`.  To install the required packages:
```bash
$ pip install -r requirements.txt
```

### Repository Structure

```
.
└── core             # Source code for the experiments
    ├── attacks            # Adversarial attacks
    ├── data               # Data setup and loading
    ├── models             # Model architectures
    └── utils              # Helpers, training and testing functions
    └── metrics.py         # Evaluation metrics
└── train.py         # Training script
└── train-wa.py      # Training with model weight averaging
└── eval-aa.py       # AutoAttack evaluation
└── eval-adv.py      # PGD+ and CW evaluation
└── eval-rb.py       # RobustBench evaluation
```

## Usage

### Training

Run [`train.py`](./train.py) for standard, adversarial, TRADES, MART and HAT training. Example commands for HAT training are provided below:

First, train a ResNet-18 model on CIFAR-10 with standard training:
```
$ python train.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc std-cifar10 \
    --data cifar10 \
    --model resnet18 \
    --num-std-epochs 50
```

Then, run the following command to perform helper-based adversarial training (HAT) on CIFAR-10:

```
$ python train.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc hat-cifar10 \
    --data cifar10 \
    --model resnet18 \
    --num-adv-epochs 50 \
    --helper-model std-cifar10 \
    --beta 2.5 \
    --gamma 0.5
```


### Robustness Evaluation

The trained models can be evaluated by running [`eval-aa.py`](./eval-aa.py) which uses [AutoAttack](https://github.com/fra31/auto-attack) for evaluating the robust accuracy. For example:
```
$ python eval-aa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc hat-cifar10
```

For evaluation with PGD+ and CW attacks, use:
```
$ python eval-adv.py --wb --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc hat-cifar10
```

### Incorporating Improvements from Gowal et al., 2020 & Rebuffi et al., 2021

HAT can be combined with imporvements from the papers "[Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples](https://arxiv.org/abs/2010.03593)" (Gowal et al., 2020) and "[Fixing Data Augmentation to Improve Adversarial Robustness](https://arxiv.org/abs/2103.01946)" (Rebuffi et al., 2021) to obtain state-of-the-art performance on multiple datasets. 


#### Training a Standard Network for Computing Helper Labels 

Train a model with standard training as [mentioned above](#training) *or* alternatively download the appropriate pretrained model from this [link](https://www.dropbox.com/sh/vzli8frhfsxo46q/AAB25dkdH6ZaDxNJzHoQNDX8a?dl=0) and place the contents of the corresponding zip file in the directory ```<log_dir>```.

#### HAT Training

Run [`train-wa.py`](./train-wa.py) for training a robust network via HAT. For example, to train a WideResNet-28-10 model via HAT on CIFAR-10 with the additional pseudolabeled data provided by [Carmon et al., 2019](https://github.com/yaircarmon/semisup-adv) or the generated datasets provided by [Rebuffi et al., 2021](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness), use the following command:

```
$ python train-wa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment> \
    --data cifar10s \
    --batch-size 1024 \
    --batch-size-validation 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.4 --tau 0.995 \
    --label-smoothing 0.1 \
    --unsup-fraction 0.7 \
    --aux-data-filename <path_to_additional_data> \
    --helper-model <helper_model_log_dir_name> \
    --beta 3.5 \
    --gamma 0.5
```


## Results

Below, we provide the results with HAT. In the settings with additional data, we follow the experimental setup used in Gowal et al., 2020 and Rebuffi et al., 2021. Whereas we resort to the experimental setup provided in our paper when not using additional data. 

##### With additional data from Carmon et al., 2019 along with the improvements by Gowal et al. 2020:

| Dataset | Norm | ε | Model | Clean Acc. | Robust Acc. |
|:---|:---:|:---:|:---:|:---:|:---:|
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8/255 | PreActResNet-18 | 89.02 | 57.67 |
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8/255 | WideResNet-28-10 | 91.30 | 62.50 |
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8/255 | WideResNet-34-10 | 91.47 | 62.83 |

Our models achieve around ~0.3-0.5% lower robustness than that reported in Gowal et al., 2020 since they use a custom regenerated pseudolabeled dataset which is not publicly available (See Section 4.3.1 [here](https://arxiv.org/abs/2010.03593)).

##### With DDPM generated data from Rebuffi et al., 2021:

| Dataset | Norm | ε | Model | CutMix | Clean Acc. | Robust Acc. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8/255 | PreActResNet-18 | &#x2717; | 86.86 | 57.09 |
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8/255 | WideResNet-28-10 | &#x2717; | 88.16 | 60.97 |
| CIFAR-10 | &#8467;<sub>2</sub> | 128/255 | PreActResNet-18 | &#x2717; | 90.57 | 76.07 |
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8/255 | PreActResNet-18 | &#x2717; | 61.50 | 28.88 |

##### Without additional data:

| Dataset | Norm | ε | Model | Clean Acc. | Robust Acc. |
|:---|:---:|:---:|:---:|:---:|:---:|
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8/255 | ResNet-18 | 84.90 | 49.08 |
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 12/255 | ResNet-18 | 77.13 | 34.56 |
| SVHN | &#8467;<sub>&infin;</sub> | 8/255 | ResNet | 92.46 | 53.19 |
| TI-200 | &#8467;<sub>&infin;</sub> | 8/255 | ResNet | 52.60 | 18.14 |


## Citing this work

```
@inproceedings{
    rade2021helperbased,
    title={Helper-based Adversarial Training: Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off},
    author={Rahul Rade and Seyed-Mohsen Moosavi-Dezfooli},
    booktitle={ICML 2021 Workshop on Adversarial Machine Learning},
    year={2021},
    url={https://openreview.net/forum?id=BuD2LmNaU3a}
}
```
