import numpy as np
import torch
from tqdm import tqdm
import libs.functions as _fn
import libs.augmentations as _aug


def train_loop(loader,
               model,
               optimizer,
               criterion,
               augmentations,
               clipping,
               num_classes,
               extras={}):
    device = next(model.parameters()).device
    losses = []
    for inputs, labels, specs in tqdm(loader):
        optimizer.zero_grad()

        bi, bl = inputs.to(device), labels.to(device)

        if augmentations['echo'] is True:
            bi, bl = _aug.echo(bi, bl, prob=.1)

        if augmentations['noise'] is True:
            bi, bl = _aug.noise(bi, bl, lib=extras['ambient'], prob=.5)

        if augmentations['augm'] == 'mix':
            bi, bl = _aug.mixup_cutmix(bi,
                                       torch.nonzero(bl, as_tuple=True)[1],
                                       num_classes)
        if augmentations['label_smoothing'] is True:
            bl = _aug.smooth_one_hot(bl, smoothing=.1)

        elif augmentations['augm'] == 'erase':
            bi, bl = _aug.erase(bi, bl)

        y_pred = model(bi)
        loss = criterion(y_pred, bl)
        losses.append(loss.item())
        loss.backward()
        if clipping:
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
    augms = [] if augmentations is None else augmentations.copy()
    augms.append('[unmodified]')
    if 'gradclip' in augms:
        grad_clip = True
        augms.remove('gradclip')
    else:
        grad_clip = False
    if 'noise' in augms:
        noise = True
        augms.remove('noise')
    else:
        noise = False
    if 'echo' in augms:
        echo = True
        augms.remove('echo')
    else:
        echo = False
    if 'label_smoothing' in augms:
        label_smoothing = True
        augms.remove('label_smoothing')
    else:
        label_smoothing = False

    _fn.report(
        f'Running training loop using augmentations: {"none" if len(augms)==1 else ", ".join(augms)} {" +echo" if echo is True else ""}{" +noise" if noise is True else ""}{" +gradient clipping" if grad_clip else ""}{" +label smoothing" if label_smoothing else ""}'
    )
    losses = []
    for augm in augms:
        losses.extend(
            train_loop(
                train_loader, model, optimizer, criterion, {
                    'augm': augm,
                    'noise': noise,
                    'echo': echo,
                    'label_smoothing': label_smoothing
                }, grad_clip, num_classes, extras))
    return losses


def validation_loop(loader, model, criterion=None):
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
            #indices
            raise Exception("Validation for indexed labels is unfinished")
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
            raise Exception(
                'Soft probabilities not implemented for validation')
        total += len(y_pred)
    model.train()
    return total, correct_t1, correct_t5, np.mean(
        losses) if len(losses) > 0 else None


def validate(train_loader, valid_loader, model, criterion):
    _fn.report("-------------- Validation ----------------")
    # retrieve non-augmented records for dataset

    total, t1, t5, _ = validation_loop(train_loader, model)
    top1train = np.round(100 * (t1 / total).cpu().numpy(), decimals=2)
    top5train = np.round(100 * (t5 / total).cpu().numpy(), decimals=2)

    total, t1, t5, loss = validation_loop(valid_loader, model, criterion)
    top1val = np.round(100 * (t1 / total).cpu().numpy(), decimals=2)
    top5val = np.round(100 * (t5 / total).cpu().numpy(), decimals=2)
    return top1train, top5train, top1val, top5val, loss
