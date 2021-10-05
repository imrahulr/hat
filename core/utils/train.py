import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.metrics import accuracy
from core.models import create_model

from .context import ctx_noparamgrad_and_eval
from .utils import seed

from .hat import at_hat_loss
from .hat import hat_loss
from .mart import mart_loss
from .rst import CosineLR
from .trades import trades_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew']


class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(Trainer, self).__init__()
        
        seed(args.seed)
        self.model = create_model(args.model, args.normalize, info, device)

        self.params = args
        self.criterion = nn.CrossEntropyLoss()
        
        self.init_optimizer(self.params.num_std_epochs)
        
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))
        
        if self.params.helper_model is not None:
            print (f'Using helper model: {self.params.helper_model}.')
            with open(os.path.join(self.params.log_dir, self.params.helper_model, 'args.txt'), 'r') as f:
                hr_args = json.load(f)
            self.hr_model = create_model(hr_args['model'], hr_args['normalize'], info, device)
            checkpoint = torch.load(os.path.join(self.params.log_dir, self.params.helper_model, 'weights-best.pt'), map_location=device)
            self.hr_model.load_state_dict(checkpoint['model_state_dict'])
            self.hr_model.eval()
            del checkpoint, hr_args
        
        self.attack, self.eval_attack = self.init_attack(self.model, self.criterion, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)
        
    
    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack,  eval_attack
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            _NUM_SAMPLES = {'svhn': 73257, 'tiny-imagenet': 100000, 'imagenet100': 128334}
            num_samples = _NUM_SAMPLES.get(self.params.data, 50000)
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            milestones = [100, 105]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=milestones)    
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            x, y = data
            x, y = x.to(device), y.to(device)
            
            if adversarial:
                if self.params.helper_model is not None and self.params.beta is not None:
                    loss, batch_metrics = self.hat_loss(x, y, h=self.params.h, beta=self.params.beta, gamma=self.params.gamma)
                elif self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                elif self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                else:
                    loss, batch_metrics = self.adversarial_loss(x, y)
            else:
                loss, batch_metrics = self.standard_loss(x, y)
                
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        return dict(metrics.mean())
    
    
    def standard_loss(self, x, y):
        """
        Standard training.
        """
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics
    
    
    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)
        
        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y
        out = self.model(x_adv)
        loss = self.criterion(out, y_adv)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item()}
        if self.params.keep_clean:
            preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
            batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds_adv)})
        else:
            batch_metrics.update({'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics
    
    
    def hat_loss(self, x, y, h, beta=1.0, gamma=1.0):
        """
        Helper-based adversarial training.
        """
        if self.params.robust_loss == 'kl':        
            loss, batch_metrics = hat_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                           epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                           h=h, beta=beta, gamma=gamma, attack=self.params.attack, hr_model=self.hr_model)
        else:
            loss, batch_metrics = at_hat_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                              epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                              h=h, beta=beta, gamma=gamma, attack=self.params.attack, hr_model=self.hr_model)
        return loss, batch_metrics
        

    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y, beta):
        """
        MART training.
        """
        loss, batch_metrics = mart_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                        epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                        beta=beta, attack=self.params.attack)
        return loss, batch_metrics  
    
    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.model(x_adv)
            else:
                out = self.model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc

    
    def save_and_eval_adversarial(self, dataloader, save, verbose=False, to_true=False, save_all=True):
        """
        Evaluate adversarial accuracy and save perturbations.
        """
        if save_all:
            x_all, y_all = [], []
        r_adv_all = []
        acc_adv = 0.0
        self.eval_attack.targeted = False
        self.model.eval()

        for x, y in tqdm(dataloader, disable=not verbose):
            x, y = x.to(device), y.to(device)
            with ctx_noparamgrad_and_eval(self.model):
                if to_true:
                    pred_y_orig = self.model(x).argmax(dim=1)
                    correct_ind = pred_y_orig == y
                    
                    x_adv, r_adv = torch.zeros(x.shape).to(device), torch.zeros(x.shape).to(device)
                    self.eval_attack.targeted = False
                    x_adv1, r_adv1 = self.eval_attack.perturb(x[correct_ind], y[correct_ind])
                    self.eval_attack.targeted = True
                    x_adv0, r_adv0 = self.eval_attack.perturb(x[~correct_ind], y[~correct_ind])
                    x_adv[correct_ind], r_adv[correct_ind]  = x_adv1, r_adv1
                    x_adv[~correct_ind], r_adv[~correct_ind] = x_adv0, r_adv0
                else:
                    x_adv, r_adv = self.eval_attack.perturb(x)
                    
            out = self.model(x_adv)
            acc_adv += accuracy(y, out)
            if save_all: 
                x_all.append(x.cpu().numpy())
                y_all.extend(y.cpu().numpy())
            r_adv_all.append((r_adv).cpu().numpy())
                
        acc_adv /= len(dataloader)
        if save:
            r_adv_all = np.vstack(r_adv_all)
            if save_all: 
                x_all = np.vstack(x_all)
                np_save({ 'x': x_all, 'r': r_adv_all, 'y': y_all }, save)
            else:
                np_save({ 'r': r_adv_all }, save)
        
        self.eval_attack.targeted = False
        return acc_adv
    
    
    def set_bn_to_eval(self):
        """
        Set all batch normalization layers to evaluation mode.
        """
        for m in self.model.modules():
            if isinstance(m, nn.modules.BatchNorm2d):
                m.eval()
    
    
    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
