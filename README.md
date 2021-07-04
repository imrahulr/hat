# Helper-based Adversarial Training

This repository contains the code for the paper "[Helper-based Adversarial Training: Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off](https://openreview.net/forum?id=BuD2LmNaU3a)" by Rahul Rade and Seyed-Mohsen Moosavi-Dezfooli. 

A short version of the paper has been accepted at the [ICML 2021 Workshop on A Blessing in Disguise: The Prospects and Perils of Adversarial Machine Learning](https://advml-workshop.github.io/icml2021/).

## Setup

### Requirements

Our code has been implemented and tested with `Python 3.7.4` and `PyTorch 1.8.0`.  To install the required packages:
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
└── train.py		 # Training script
└── train-wa.py		 # Training with model weight averaging
└── eval-aa.py		 # AutoAttack evaluation
└── eval-adv.py		 # PGD+ and CW evaluation
└── eval-rb.py		 # RobustBench evaluation
```

## Usage

### Training

Run [`train.py`](./train.py) for standard, adversarial, TRADES, MART and HAT training. Example commands for HAT training are provided below:

Fisrt, train a ResNet-18 model on CIFAR-10 with standard training:
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

The trained models can be evaluated by running [eval-aa.py](./eval-aa.py) which uses AutoAttack for evaluating the robust accuracy.
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

### Incorporating Improvements from Gowal et al., 2020

HAT can be combined with the imporvements from the paper "[Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples](https://arxiv.org/abs/2010.03593)"  (Gowal et al., 2020) to obtain state-of-the-art performance on CIFAR-10. 


##### Training a Standard Network for Computing Helper Labels 

Train a model on CIFAR-10 with standard training as [mentioned above](#training) *or* alternatively download the pretrained model from this [link](https://www.dropbox.com/sh/vzli8frhfsxo46q/AAB25dkdH6ZaDxNJzHoQNDX8a?dl=0) and place the contents of the corresponding zip file in the ```<log_dir>```.

##### HAT Training

Run [`train-wa.py`](./train-wa.py) for training a robust network via HAT along with the imporvements from Gowal et al., 2020. For example, train a WideResNet-28-10 model via HAT on CIFAR-10 with the additional pseudolabeled data provided by [Carmon et al., 2019](https://github.com/yaircarmon/semisup-adv):

```
$ python train-wa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment> \
    --data cifar10s \
    --batch-size 1024 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.4 --tau 0.995 \
    --unsup-fraction 0.7 \
    --aux-data-filename <path_to_additional_data> \
    --helper-model <helper_model_log_dir_name> \
    --beta 3.5 \
    --gamma 0.5
```



## Reference

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