import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.attacks import create_attack
from core.metrics import accuracy
from core.utils.context import ctx_noparamgrad_and_eval
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats


def hat_loss(model, x, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, h=3.5, beta=1.0, gamma=1.0, 
             attack='linf-pgd', hr_model=None, label_smoothing=0.1):
    """
    TRADES + Helper-based adversarial training.
    """
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.train()
    track_bn_stats(model, False)
    
    x_adv = x.detach() +  torch.FloatTensor(x.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    p_natural = F.softmax(model(x), dim=1).detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack == 'l2-pgd':
        delta = torch.FloatTensor(x.shape).normal_(mean=0, std=1.0).cuda().detach()
        delta.data = delta.data * np.random.uniform(0.0, epsilon) / (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
        delta = Variable(delta.data, requires_grad=True).cuda()
        
        batch_size = len(x)
        optimizer_delta = torch.optim.SGD([delta], lr=step_size)
        for _ in range(perturb_steps):
            adv = x + delta
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)
          
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_hr = x + h * (x_adv - x)
    with ctx_noparamgrad_and_eval(hr_model):
        y_hr = hr_model(x_adv).argmax(dim=1) 
        
    optimizer.zero_grad()
    track_bn_stats(model, True)
    
    # a hack to save memory when using large batch sizes.
    # first, calculate gradients with clean and adversarial samples.
    # then, clear intermediate activations and calculate gradients with helper samples.
    # one can use a single .backward() at the expense of higher memory usage.
    out_clean = model(x)
    out_adv = model(x_adv)
    loss_clean = criterion_ce(out_clean, y)
    loss_adv = (1/len(x)) * criterion_kl(F.log_softmax(out_adv, dim=1), F.softmax(out_clean, dim=1))
    loss = loss_clean + beta * loss_adv
    total_loss = loss.item()
    loss.backward()
    
    out_help = model(x_hr)
    loss = gamma * F.cross_entropy(out_help, y_hr, reduction='mean')
    total_loss += loss.item()
    
    batch_metrics = {'loss': total_loss}
    batch_metrics.update({'adversarial_acc': accuracy(y, out_adv.detach()), 'helper_acc': accuracy(y_hr, out_help.detach())}) 
    batch_metrics.update({'clean_acc': accuracy(y, out_clean.detach())})
    return loss, batch_metrics


def at_hat_loss(model, x, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, h=3.5, beta=1.0, gamma=1.0, 
                attack='linf-pgd', hr_model=None, label_smoothing=0.1):
    """
    AT + Helper-based adversarial training.
    """
    criterion_ce_smooth = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_ce = nn.CrossEntropyLoss()
    model.train()
    track_bn_stats(model, False)
    
    attack = create_attack(model, criterion_ce, attack, epsilon, perturb_steps, step_size)
    with ctx_noparamgrad_and_eval(model):
        x_adv, _ = attack.perturb(x, y)
        
    model.train()
    
    x_hr = x + h * (x_adv - x)
    with ctx_noparamgrad_and_eval(hr_model):
        y_hr = hr_model(x_adv).argmax(dim=1)
  
    optimizer.zero_grad()
    track_bn_stats(model, True)
    
    out_clean, out_adv, out_help = model(x), model(x_adv), model(x_hr)
    loss_clean = criterion_ce_smooth(out_clean, y)
    loss_adv = criterion_ce(out_adv, y)
    loss_help = F.cross_entropy(out_help, y_hr, reduction='mean')
    loss = loss_clean + beta * loss_adv + gamma * loss_help
     
    batch_metrics = {'loss': loss.item()}
    batch_metrics.update({'adversarial_acc': accuracy(y, out_adv.detach()), 'helper_acc': accuracy(y_hr, out_help.detach())}) 
    batch_metrics.update({'clean_acc': accuracy(y, out_clean.detach())})
    return loss, batch_metrics
