import numpy as np
import torch
from tqdm import tqdm
import libs.functions as _fn
import libs.augmentations as _aug


def train_loop(loader,
               model,
               optimizer,
               criterion,
               device,
               augmentation,
               clipping,
               num_classes,
               extras={}):
    losses = []
    for inputs, labels, specs in tqdm(loader):
        optimizer.zero_grad()
        if augmentation == 'mixup':
            bi, bl = _aug.mixup(
                inputs.to(device),
                torch.nonzero(labels, as_tuple=True)[1].to(device),
                num_classes)
        elif augmentation == 'cutmix':
            bi, bl = _aug.cutmix(
                inputs.to(device),
                torch.nonzero(labels, as_tuple=True)[1].to(device),
                num_classes)
        elif augmentation == 'erase':
            bi, bl = _aug.erase(inputs.to(device), labels.to(device))
        elif augmentation == 'whitenoise':
            bi, bl = _aug.whitenoise(inputs.to(device), labels.to(device))
        elif augmentation == 'ambient_noise':
            bi, bl = _aug.ambient_noise(inputs.to(device), labels.to(device),
                                        extras['ambient'])
        elif augmentation == 'music_noise':
            bi, bl = _aug.music_noise(inputs.to(device), labels.to(device),
                                      extras['music'])
        else:
            bi, bl = inputs.to(device), labels.to(device)
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
          device,
          num_classes,
          augmentations=None,
          extras={}):
    if augmentations is None:
        augmentations = []
    augms = ['_']
    augms.extend(augmentations)
    if 'gradclip' in augms:
        grad_clip = True
        augms.remove('gradclip')
    else:
        grad_clip = False
    _fn.report(
        f'Running train loop with following augmentations: {"None" if len(augms)==1 else augms}, using gradient clipping: {grad_clip}'
    )
    losses = []
    for augm in augms:
        losses.extend(
            train_loop(train_loader, model, optimizer, criterion, device, augm,
                       grad_clip, num_classes, extras))
    return losses


def validation_loop(loader, model, device):
    total, correct_t1, correct_t5 = 0, 0, 0
    model.eval()
    for inputs, labels, _ in tqdm(loader):
        with torch.no_grad():
            y_pred = model(inputs.to(device))
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
    return total, correct_t1, correct_t5


def validate(train_loader, valid_loader, model, device):
    _fn.report("-------------- Validation ----------------")
    # retrieve non-augmented records for dataset

    total, t1, t5 = validation_loop(train_loader, model, device)
    top1train = np.round(100 * (t1 / total).cpu().numpy(), decimals=2)
    top5train = np.round(100 * (t5 / total).cpu().numpy(), decimals=2)

    total, t1, t5 = validation_loop(valid_loader, model, device)
    top1val = np.round(100 * (t1 / total).cpu().numpy(), decimals=2)
    top5val = np.round(100 * (t5 / total).cpu().numpy(), decimals=2)
    '''
    D.reset()
    valCandidates = D.get_random_records(limit_per_speaker=10, flag=False)
    top5train = np.round(
        100 *
        top5(candidates=[np.squeeze(el['embedding']) for el in valCandidates],
             centroids=centroids,
             truths=[el['speaker'] for el in valCandidates]),
        decimals=2)

    _fn.report("Top 1 accuracy dist(sample, centroids): " + str(top1train) +
               "%")
    _fn.report("Top 5 accuracy dist(sample, centroids): " + str(top5train) +
               "%")

    V.reset()
    valCandidates = V.get_random_records(limit_per_speaker=10, flag=False)
    top1val = np.round(
        100 *
        top1(candidates=[np.squeeze(el['embedding']) for el in valCandidates],
             centroids=centroids,
             truths=[el['speaker'] for el in valCandidates]),
        decimals=2)
    top5val = np.round(
        100 *
        top5(candidates=[np.squeeze(el['embedding']) for el in valCandidates],
             centroids=centroids,
             truths=[el['speaker'] for el in valCandidates]),
        decimals=2)
    _fn.report("Top 1 val accuracy dist(sample, centroids): " + str(top1val) +
               "%")
    _fn.report("Top 5 val accuracy dist(sample, centroids): " + str(top5val) +
               "%")
    return top1train, top5train, top1val, top5val
    '''
    return top1train, top5train, top1val, top5val
