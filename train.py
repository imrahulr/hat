"""
Standard Training + Adversarial Training.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train
from core.utils import Trainer
from core.utils import seed
from core import setup



# Setup

parse = parser_train()
args = parse.parse_args()


DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
TMP = os.path.join(args.tmp_dir, args.desc)
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
if args.exp and not os.path.exists(TMP):
    os.makedirs(TMP, exist_ok=True)
    print ('Tmp Dir: ', TMP)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

if 'imagenet' in args.data:
    setup.setup_train(DATA_DIR)
    setup.setup_val(DATA_DIR)
    args.data_dir = os.environ['TMPDIR']
    DATA_DIR = os.path.join(args.data_dir, args.data)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_STD_EPOCHS = args.num_std_epochs
NUM_ADV_EPOCHS = args.num_adv_epochs
NUM_SAMPLES_EVAL = args.num_samples_eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_STD_EPOCHS = 1
    NUM_ADV_EPOCHS = 1

# To speed up training
if args.model in ['wrn-34-10', 'wrn-34-20'] or 'swish' in args.model or 'imagenet' in args.data:
    torch.backends.cudnn.benchmark = True


# Load data

seed(args.seed)
train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction
)
num_train_samples = len(train_dataset)
num_test_samples = len(test_dataset)

train_indices = np.random.choice(num_train_samples, NUM_SAMPLES_EVAL, replace=False)
test_indices = np.random.choice(num_test_samples, NUM_SAMPLES_EVAL, replace=False)

pin_memory = torch.cuda.is_available()
if args.exp:
    train_eval_dataset = torch.utils.data.Subset(train_dataset, train_indices[:NUM_SAMPLES_EVAL])
    train_eval_dataloader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=BATCH_SIZE_VALIDATION, shuffle=False, 
                                                        num_workers=4, pin_memory=pin_memory)

    test_eval_dataset = torch.utils.data.Subset(test_dataset, test_indices[:NUM_SAMPLES_EVAL])
    test_eval_dataloader = torch.utils.data.DataLoader(test_eval_dataset, batch_size=BATCH_SIZE_VALIDATION, shuffle=False, 
                                                       num_workers=4, pin_memory=pin_memory)
    del train_eval_dataset, test_eval_dataset
del train_dataset, test_dataset



# Standard Training

seed(args.seed)
metrics = pd.DataFrame()
trainer = Trainer(info, args)
last_lr = args.lr

logger.log('\n\n')
logger.log('Standard training for {} epochs'.format(NUM_STD_EPOCHS))
old_score = [0.0]

for epoch in range(1, NUM_STD_EPOCHS+1):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    res = trainer.train(train_dataloader, epoch=epoch)
    test_acc = trainer.eval(test_dataloader)
    
    if test_acc >= old_score[0]:
        old_score[0] = test_acc
        trainer.save_model(WEIGHTS)

    logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
    logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})
    
    if epoch == NUM_STD_EPOCHS:
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        logger.log('Adversarial Accuracy-\tTest: {:.2f}%.'.format(test_adv_acc*100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
    metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
        
if NUM_STD_EPOCHS > 0:
    trainer.load_model(WEIGHTS)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats_std.csv'), index=False)

    

# Adversarial Training (AT, TRADES, MART and HAT)

if NUM_ADV_EPOCHS > 0:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)*100))
    
    if args.exp:
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        logger.log('Adversarial Accuracy-\tTest: {:2f}%.'.format(test_adv_acc*100))
        trainer.save_model(os.path.join(TMP, 'model_0.pt'))     
        _ = trainer.save_and_eval_adversarial(train_eval_dataloader, save=os.path.join(TMP, 'eval_train_adv_0'))
        _ = trainer.save_and_eval_adversarial(test_eval_dataloader, save=os.path.join(TMP, 'eval_test_adv_0'))
    
    old_score = [0.0, 0.0]
    logger.log('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0    
    

for epoch in range(1, NUM_ADV_EPOCHS+1):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
    test_acc = trainer.eval(test_dataloader)

    if args.exp and (epoch % 5 == 0 or epoch == 1):
        trainer.save_model(os.path.join(TMP, 'model_{}.pt'.format(epoch)))
        save_eval_train_file = os.path.join(TMP, 'eval_train_adv_{}'.format(epoch))
        save_eval_test_file = os.path.join(TMP, 'eval_test_adv_{}'.format(epoch))
        _ = trainer.save_and_eval_adversarial(train_eval_dataloader, save=save_eval_train_file, save_all=False)
        _ = trainer.save_and_eval_adversarial(test_eval_dataloader, save=save_eval_test_file, save_all=False)
    
    logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
    if 'clean_acc' in res:
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
    else:
        logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({'epoch': NUM_STD_EPOCHS+epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})
    
    if epoch % args.adv_eval_freq == 0 or epoch > (NUM_ADV_EPOCHS-5) or (epoch >= (NUM_ADV_EPOCHS-10) and NUM_ADV_EPOCHS > 90):
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, 
                                                                                   test_adv_acc*100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
    else:
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))
    
    if test_adv_acc >= old_score[1]:
        old_score[0], old_score[1] = test_acc, test_adv_acc
        trainer.save_model(WEIGHTS)
    trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))

    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)

    
    
# Record metrics

train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
logger.log('\nTraining completed.')
logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
if NUM_ADV_EPOCHS > 0:
    logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

logger.log('Script Completed.')
