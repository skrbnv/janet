import torch
#import torchvision.transforms as transforms
import numpy as np
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
#from timm.data.transforms import RandomResizedCropAndInterpolation
#import warnings
#import librosa


def erase(inputs, classes):
    erase_fn = RandomErasing(probability=1)
    return erase_fn(inputs), classes


def mixup(inputs, classes, num_classes):
    mixup_args = {
        'mixup_alpha': 1.,
        'cutmix_alpha': 0.,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': num_classes
    }
    mixup_fn = Mixup(**mixup_args)
    return mixup_fn(inputs, classes)


def cutmix(inputs, classes, num_classes):
    cutmix_args = {
        'mixup_alpha': 0.,
        'cutmix_alpha': 1.0,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': num_classes
    }
    cutmix_fn = Mixup(**cutmix_args)
    return cutmix_fn(inputs, classes)


def whitenoise(inputs, classes, mean=0, std=5e-3):
    return inputs + (torch.randn(inputs.shape) * std + mean).to(
        inputs.device), classes


def ambient_noise(inputs, classes, lib, amplif=1):
    return lib.batch_overlay(inputs, amplif), classes


def music_noise(inputs, classes, lib, amplif=1):
    return lib.batch_overlay(inputs, amplif), classes
