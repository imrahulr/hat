import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import torch

from core.data.utils import AdversarialDatasetWithPerturbation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_orthogonal_vector(r):
    """
    Returns a random unit vector orthogonal to given unit vector r.
    """
    r = r / torch.norm(r.view(-1), p=2)
    p = torch.rand(r.numel()).to(device)
    p = p - p.dot(r.view(-1))*r.view(-1)
    p = p / torch.norm(p, p=2)
    p = p.view(r.shape)
    assert np.isclose(torch.dot(p.view(-1), r.view(-1)).item(), 0, atol=1e-6) == True, 'p and r are not orthogonal.'
    return p


def line_search(model, x, r, y, precision=0.1, ord=2, max_alpha=35, normalize_r=True, clip_min=0., clip_max=1., ortho=False):
    """
    Perform line search to find margin.
    """
    x, r = x.unsqueeze(0), r.unsqueeze(0)
    pert_preds = model(torch.clamp(x+r, 0, 1))
    
    if normalize_r:
        r = r / r.view(-1).norm(p=ord)
    if ortho:
        r = get_orthogonal_vector(r)
       
    orig_preds = model(x)
    orig_labels = orig_preds.argmax(dim=1)
    pert_x = replicate_input(x)
    for a in range(0, max_alpha + 1): # fast search
        pert_labels = model(pert_x).argmax(dim=1)
        if pert_labels != orig_labels:
            break
        pert_x = x + a*r
        alpha = a
    
    pert_x = replicate_input(x)
    if alpha != max_alpha: # fine-tune search with given precision
        for a in np.arange(alpha - 1, alpha + precision, precision):
            pert_labels = model(pert_x).argmax(dim=1)
            if pert_labels != orig_labels:
                break
            pert_x = x + a*r
        margin = a
    else:
        margin = max_alpha
    
    pert_labels = pert_preds.argmax(dim=1)
    return {'mar': margin, 'true': y, 'orig_pred': orig_labels.item(), 'pert_pred': pert_labels.item()}


def measure_margin(trainer, data_path, precision, ord=2, ortho=False, verbose=False):    
    """
    Estimate margin using line search.
    """
    
    if ord not in [2, np.inf]:
        raise NotImplementedError('Only ord=2 and ord=inf have been implemented!')
    trainer.model.eval()

    mar_adv_any = []
    dataset = AdversarialDatasetWithPerturbation(data_path)
    for x, r, y in tqdm(dataset, disable=not verbose):
        x, r = x.to(device), r.to(device)
        mar_any = line_search(trainer.model, x, r, y, ord=ord, precision=precision, ortho=ortho)    
        mar_adv_any.append(mar_any)  
    assert len(mar_adv_any) == len(dataset), 'Lengths must match'
    
    mar_adv_any = pd.DataFrame(mar_adv_any)
    mar10, mar50, mar90 = np.percentile(mar_adv_any['mar'], [10, 50, 90])
    out_margin = {'mean_margin': np.mean(mar_adv_any['mar']), '10_margin': mar10, '50_margin': mar50, '90_margin': mar90} 

    return mar_adv_any, out_margin
