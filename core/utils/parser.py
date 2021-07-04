import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from .train import SCHEDULERS

from .utils import str2bool, str2float


def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description='Standard + Adversarial Training.')

    parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=256, help='Batch size for testing.')
    parser.add_argument('--num-samples-eval', type=int, default=512, help='Number of samples to use for margin calculations.')
    
    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/adversarial-hat/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/home/rarade/adversarial-hat/logs/')
    parser.add_argument('--tmp-dir', type=str, default='/cluster/scratch/rarade/')
    
    parser.add_argument('-d', '--data', type=str, default='cifar10', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str, required=True, 
                        help='Description of experiment. It will be used to name directories.')

    parser.add_argument('-m', '--model', choices=MODELS, default='resnet18', help='Model architecture to be used.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('-ns', '--num-std-epochs', type=int, default=0, help='Number of standard training epochs.')
    parser.add_argument('-na', '--num-adv-epochs', type=int, default=0, help='Number of adversarial training epochs.')
    parser.add_argument('--adv-eval-freq', type=int, default=30, help='Adversarial evaluation frequency (in epochs).')
    
    parser.add_argument('--h', default=2.0, type=float, help='Parameter h to compute helper examples (x + h*r) for HAT.')
    parser.add_argument('--helper-model', type=str, default=None, help='Helper model weights file name for HAT.')
    parser.add_argument('--beta', default=None, type=float, help='Stability regularization, i.e., 1/lambda in TRADES \
    or weight of robust loss in HAT.')
    parser.add_argument('--gamma', default=1.0, type=float, help='Weight of helper loss in HAT.')
    parser.add_argument('--robust-loss', default='kl', choices=['ce', 'kl'], type=str, help='Type of robust loss in HAT.')
    
    parser.add_argument('--lr', type=float, default=0.21, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cyclic', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--keep-clean', type=str2bool, default=False, help='Use clean samples during adversarial training.')

    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')
    parser.add_argument('--exp', action='store_true', default=False, 
                        help='Store results for performing margin and curvature experiments later.')
    parser.add_argument('--mart', action='store_true', default=False, help='MART training.')
    
    parser.add_argument('--unsup-fraction', type=float, default=0.5, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux-data-filename', type=str, help='Path to additional Tiny Images data.', 
                        default='/cluster/scratch/rarade/cifar10s/ti_500K_pseudo_labeled.pickle')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser


def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/adversarial-hat/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/home/rarade/adversarial-hat/logs/')
    parser.add_argument('--tmp-dir', type=str, default='/cluster/scratch/rarade/')
        
    parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')
    
    # eval-aa.py
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')

    # eval-adv.py
    parser.add_argument('--source', type=str, default=None, help='Path to source model for black-box evaluation.')
    parser.add_argument('--wb', action='store_true', default=False, help='Perform white-box PGD evaluation.')
    
    # eval-rb.py
    parser.add_argument('--threat', type=str, default='corruptions', choices=['corruptions', 'Linf', 'L2'],
                        help='Threat model for RobustBench evaluation.')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser

