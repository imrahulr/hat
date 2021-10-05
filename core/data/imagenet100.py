import numpy as np

import torchvision.transforms as transforms
from robustness.datasets import CustomImageNet


DATA_DESC = {
    'data': 'imagenet100',
    'classes': np.arange(100),
    'num_classes': 100,
    'mean': [0.485, 0.456, 0.406], 
    'std': [0.229, 0.224, 0.225],
}


class ImageNet100(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)] if '100' not in data_path else 
                            [[label] for label in range(0, 100)],
            **kwargs,
        )

def load_imagenet100(data_dir, use_augmentation=False):
    """
    Returns ImageNet100 train, test datasets.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    temp_dataset = ImageNet100(data_dir)
    train_dataloader, test_dataloader = temp_dataset.make_loaders(4, 128)
    
    train_dataset = train_dataloader.dataset
    train_dataset.transform = train_transform
    test_dataset = test_dataloader.dataset
    test_dataset.transform = test_transform
    return train_dataset, test_dataset