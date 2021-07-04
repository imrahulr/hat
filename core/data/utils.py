import os
import re
import numpy as np

import torch

from core.utils.utils import np_load
from core.utils.utils import NumpyToTensor
    
    
class AdversarialDatasetWithPerturbation(torch.utils.data.Dataset):
    """
    Torch dataset for reading examples with corresponding perturbations.
    Arguments:
        root (str): path to saved data.
        transform (torch.nn.Module): transformations to be applied to input.
        target_transform (torch.nn.Module): transformations to be applied to target.
    """
    def __init__(self, root, transform=NumpyToTensor(), target_transform=None):
        super(AdversarialDatasetWithPerturbation, self).__init__()
        
        x_path = re.sub(r'adv_(\d)+', 'adv_0', root)   
        if os.path.isfile(os.path.join(root, 'x.npy')):
            data = np_load(x_path)
        elif os.path.isfile(os.path.join(x_path, 'x.npy')):
            data = np_load(x_path)
        else:
            raise FileNotFoundError('x, y not found at {} and {}.'.format(root, x_path))
        self.data = data['x']
        self.targets = data['y']
        
        data = np_load(root)
        self.r = data['r']
        self.transform = transform
        self.target_transform = target_transform
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
            r = self.transform(self.r[idx])
        if self.target_transform:
            label = self.target_transform(label)
        return image, r, label
