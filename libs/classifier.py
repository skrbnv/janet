import numpy as np
import torch
from tqdm import tqdm
import libs.functions as _fn
import libs.augmentations as _aug
import random


def train_loop(train_loader,
               model,
               optimizer,
               criterion,
               num_classes,
               augmentations=None,
               extras={},
               lsm=False,
               gclip=False,
               config={
                   'mixprob': 1.,
                   'eraseprob': 1.,
               }):
    device = next(model.parameters()).device
    losses = []
    for inputs, labels, _ in (progressbar := tqdm(train_loader)):
        optimizer.zero_grad()
        bi, bl = inputs.to(device), labels.to(device)
        augm = random.choice(augmentations) if len(augmentations) > 0 else ''
        if augm == 'echo':
            bi, bl = _aug.echo(bi, bl, prob=1)
        elif augm == 'noise':
            bi, bl = _aug.noise(bi, bl, lib=extras['ambient'], prob=1)
        elif augm == 'mix':
            bi, bl = _aug.mixup_cutmix(bi,
                                       torch.nonzero(bl, as_tuple=True)[1],
                                       num_classes, config['mixprob'])
        elif augm == 'erase':
            bi, bl = _aug.erase(bi, bl, config['eraseprob'])
        if lsm is True:
            bl = _aug.smooth_one_hot(bl, smoothing=.1)

        y_pred = model(bi)
        loss = criterion(y_pred, bl)
        losses.append(loss.item())
        loss.backward()
        ins = '/'.join(augmentations) if len(augmentations) > 0 else '  '
        progressbar.set_description(f'[{ins}] {loss.item():.4f}')
        if gclip is True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
    return losses


def train(train_loader,
          model,
          optimizer,
          criterion,
          num_classes,
          augmentations=None,
          extras={}):

    augms = augmentations[:] if augmentations is not None else []
    if 'label_smoothing' in augms:
        lsm = True
        augms.remove('label_smoothing')
    else:
        lsm = False
    if 'grad_clip' in augms:
        gclip = True
        augms.remove('grad_clip')
    else:
        gclip = False
    augms.append('')
    losses = []
    for augm in augms:
        losses.extend(
            train_loop(train_loader=train_loader,
                       model=model,
                       optimizer=optimizer,
                       criterion=criterion,
                       num_classes=num_classes,
                       augmentations=[augm],
                       extras=extras,
                       lsm=lsm,
                       gclip=gclip))

    return losses


def fasttrain(train_loader,
              model,
              optimizer,
              criterion,
              num_classes,
              augmentations=None,
              extras={}):

    augms = augmentations[:]
    if 'label_smoothing' in augms:
        lsm = True
        augms.remove('label_smoothing')
    else:
        lsm = False
    if 'grad_clip' in augms:
        gclip = True
        augms.remove('grad_clip')
    else:
        gclip = False

    losses = []
    if len(augms) > 0:
        losses.extend(
            train_loop(train_loader=train_loader,
                       model=model,
                       optimizer=optimizer,
                       criterion=criterion,
                       num_classes=num_classes,
                       augmentations=augms,
                       extras=extras,
                       lsm=lsm,
                       gclip=gclip,
                       config={
                           'mixprob': 0.5,
                           'eraseprob': 0.5,
                       }))
    return losses


def test_loop(loader, model, criterion=None):
    device = next(model.parameters()).device
    total, correct_t1, correct_t5 = 0, 0, 0
    model.eval()
    losses = []
    for inputs, labels, _ in tqdm(loader):
        with torch.no_grad():
            y_pred = model(inputs.to(device))
            if criterion is not None:
                loss = criterion(y_pred, labels.to(device))
                losses.append(loss.item())
        # three options: indices, one-hot or soft probabilities (including n-hot)
        if len(labels[0]) == 1:
            # indices
            raise Exception("Not implemented")
            indices_t1 = torch.argmax(y_pred, dim=1)
            indices_t5 = torch.topk(y_pred, 5)
            correct_t1 += torch.sum(indices_t1 == labels.to(device))
        elif torch.sum(labels == 1) == len(labels):
            # onehot
            # convert to indices
            indices_true = torch.nonzero(labels, as_tuple=True)[1].to(device)
            indices_t1 = torch.argmax(y_pred, dim=1)
            indices_t5 = torch.topk(y_pred, 5, dim=1)[1]
            correct_t5 += torch.sum(indices_t5 == torch.broadcast_to(
                indices_true.unsqueeze(1), indices_t5.shape))
            correct_t1 += torch.sum(indices_t1 == indices_true)
        else:
            raise Exception('Soft probabilities not implemented')
        total += len(y_pred)
    model.train()
    return total, correct_t1, correct_t5, np.mean(
        losses) if len(losses) > 0 else None


def test(train_loader, test_loader, model, criterion):
    _fn.report("-------------- Testing ----------------")
    # retrieve non-augmented records for dataset

    total, t1, t5, _ = test_loop(train_loader, model)
    top1train = np.round(100 * (t1 / total).cpu().numpy(), decimals=2)
    top5train = np.round(100 * (t5 / total).cpu().numpy(), decimals=2)

    total, t1, t5, loss = test_loop(test_loader, model, criterion)
    top1test = np.round(100 * (t1 / total).cpu().numpy(), decimals=2)
    top5test = np.round(100 * (t5 / total).cpu().numpy(), decimals=2)
    return top1train, top5train, top1test, top5test, loss
