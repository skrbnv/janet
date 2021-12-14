import torch
import torchvision.transforms as transforms
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import RandomResizedCropAndInterpolation
import warnings


def erase(inputs, classes):
    erase_fn = RandomErasing(probability=1)
    return erase_fn(inputs), classes


def mixup(inputs, classes):
    mixup_args = {
        'mixup_alpha': 1.,
        'cutmix_alpha': 0.,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': 630
    }
    mixup_fn = Mixup(**mixup_args)
    warnings.warn('Running mixup operation with 630 classes')
    return mixup_fn(inputs, classes)


def cutmix(inputs, classes):
    cutmix_args = {
        'mixup_alpha': 0.,
        'cutmix_alpha': 1.0,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': 630
    }
    cutmix_fn = Mixup(**cutmix_args)
    warnings.warn('Running cutmix operation with 630 classes')
    return cutmix_fn(inputs, classes)


def scale(inputs, classes):
    toPIL = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    scale_fn = RandomResizedCropAndInterpolation(size=(64, 192),
                                                 scale=(0.9, 1.1),
                                                 ratio=(1. / 3.))
    outputs = [toTensor(scale_fn(toPIL(el))) for el in inputs]
    return torch.stack(outputs).to(inputs.device), classes
