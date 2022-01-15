import torch
import torchvision.transforms as transforms
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import RandomResizedCropAndInterpolation
import warnings
import librosa


def identity(inputs, classes):
    return inputs, classes


def pitch_shift(inputs, classes, scale=(0.968, 1.)):
    # crop along y axis [0 to XX%] (notice 0)
    # then resize to target size
    assert len(scale) == 2, ValueError(
        "Scale factor must be tuple of two items: min crop and max crop")
    assert len(
        inputs.shape) == 4, ValueError("inputs should have 4 dim: BxCxHxW")
    outputs = torch.empty(inputs.shape).to(inputs.device)
    r1, r2 = scale[0], scale[1]
    sv = ((torch.rand(len(inputs)) * (r2 - r1) + r1) *
          inputs[0].shape[1]).int()
    resize_fn = transforms.Resize((inputs.shape[-2], inputs.shape[-1]))
    for i, input in enumerate(inputs):
        cropped = input[:, :sv[i], :]
        outputs[i] = resize_fn(cropped)
    return outputs, classes


def speed(inputs, classes, scale=(0.948, 1.)):
    # crop along y axis [0 to XX%] (notice 0)
    # then resize to target size
    assert len(scale) == 2, ValueError(
        "Scale factor must be tuple of two items: min crop and max crop")
    assert len(
        inputs.shape) == 4, ValueError("inputs should have 4 dim: BxCxHxW")
    outputs = torch.empty(inputs.shape).to(inputs.device)
    r1, r2 = scale[0], scale[1]
    sv = ((torch.rand(len(inputs)) * (r2 - r1) + r1) *
          inputs[0].shape[2]).int()
    resize_fn = transforms.Resize((inputs.shape[-2], inputs.shape[-1]))
    for i, input in enumerate(inputs):
        cropped = input[:, :, :sv[i]]
        outputs[i] = resize_fn(cropped)
    return outputs, classes


def volume(inputs, classes, scale=(0.9, 1.1)):
    assert len(scale) == 2, ValueError(
        "Scale factor must be tuple of two items: min crop and max crop")
    assert len(
        inputs.shape) == 4, ValueError("inputs should have 4 dim: BxCxHxW")
    r1, r2 = scale[0], scale[1]
    mults = (torch.rand(len(inputs)) * (r2 - r1) + r1).to(inputs.device)
    shape = len(inputs.shape) * [1]
    shape[0] = len(mults)
    return inputs * mults.reshape(tuple(shape)), classes


def contrast(inputs, classes, scale=(0.8, 1.2)):
    assert len(scale) == 2, ValueError(
        "Scale factor must be tuple of two items: min crop and max crop")
    assert len(
        inputs.shape) == 4, ValueError("inputs should have 4 dim: BxCxHxW")
    outputs = torch.empty(inputs.shape).to(inputs.device)
    r1, r2 = scale[0], scale[1]
    mults = torch.rand(len(inputs)) * (r2 - r1) + r1
    contrast_fn = transforms.functional.adjust_contrast
    for i, input in enumerate(inputs):
        outputs[i] = contrast_fn(input, mults[i])
    return outputs, classes


def equalize(inputs, classes):
    assert len(
        inputs.shape) == 4, ValueError("inputs should have 4 dim: BxCxHxW")
    fn_pre = transforms.ToPILImage()
    fn_post = transforms.PILToTensor()
    fn = transforms.functional.equalize
    outputs = torch.empty((inputs.shape)).to(inputs.device)
    for i, input in enumerate(inputs):
        outputs[i] = fn_post(fn(fn_pre((input + 1.) / 2.))) / 255. * 2. - 1.
    return outputs, classes


def whitenoise(inputs,
               classes,
               rms,
               snr=15,
               sr=16000,
               n_fft=400,
               fmin=20,
               fmax=8000,
               htk=True,
               n_mels=64):
    assert len(
        inputs.shape) == 4, ValueError("inputs should have 4 dim: BxCxHxW")
    outputs = torch.empty((inputs.shape)).to(inputs.device)
    for i, input in enumerate(inputs):
        noise = torch.normal(0,
                             torch.sqrt(rms**2 / 10**(snr / 10)),
                             size=(input.shape[0] * 160))
        noise = (torch.abs(
            librosa.stft(y=noise,
                         n_fft=400,
                         hop_length=160,
                         window='hamming',
                         center=True))**2)[:, :input.shape[0]]
        mel_basis = librosa.filters.mel(sr=sr,
                                        n_fft=n_fft,
                                        fmin=fmin,
                                        fmax=fmax,
                                        htk=True,
                                        n_mels=n_mels)
        mel_noise = torch.dot(mel_basis, noise)
        back_one = librosa.db_to_power(input)
        outputs[i] = librosa.power_to_db(back_one + mel_noise.T)
    return outputs, classes


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
