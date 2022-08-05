import numpy as np
import torch
from libs.datasets import TripletDataset
import multiprocessing as mp
from libs.multiprocessing import set_affinity_on_worker
from copy import deepcopy


def sub_generate_triplet_pair(D,
                              anchor,
                              positive_pool,
                              negative_pool,
                              margin=.3,
                              positive_strategy='Random',
                              negativesemihard=True,
                              negativehard=True):
    if positive_strategy == 'Hard':
        positive = D.get_farthest_sibling(anchor['_id'])
        negative = D.get_closest_opposite(anchor['_id'], limit_scope=100)
        return anchor, positive, negative

    elif positive_strategy == 'Random':
        for positive in positive_pool:
            if positive['_id'] == anchor['_id']:
                continue
            dAP = D.distance(anchor['_id'], positive['_id'])

            np.random.shuffle(negative_pool)
            for negative in negative_pool:
                dAN = D.distance(anchor['_id'], negative['_id'])
                # semi hard selection: dAP < dAN < dAP+margin
                # hard selection: dAN < dAP
                if (negativesemihard and dAP < dAN
                        and dAN < dAP + margin) or (negativehard
                                                    and dAP >= dAN):
                    return anchor, positive, negative

    return None, None, None


def generate_triplets_mp(D=None,
                         limit=64,
                         positive_strategy='Random',
                         negativesemihard=True,
                         negativehard=True):
    if D is None:
        raise Exception('No dataset provided for triplets generation')
    aa, pp, nn = [], [], []
    speakers = D.get_unique_speakers()
    #_fn.report(f'-- Extracted {len(speakers)} speakers')
    D.calculate_distances()
    #_fn.report('-- Distances calculated')

    pool = mp.Pool(processes=mp.cpu_count(),
                   initializer=set_affinity_on_worker)
    results = []
    for speaker in speakers:
        result = pool.apply_async(
            generate_triplets, (deepcopy(D), limit, speaker, positive_strategy,
                                negativesemihard, negativehard))
        results.append(result)
    while True:
        #sleep(1)
        #print('\u263d', end='')
        try:
            ready = [result.ready() for result in results]
            #print(f'{sum(ready)} out of {len(ready)}')
            successful = [result.successful() for result in results]
        except Exception:
            continue
        if all(successful):
            #print('')
            break
        if all(ready) and not all(successful):
            raise Exception(
                f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}'
            )
    for el in [result._value for result in results]:
        a, p, n = el
        aa.extend(a)
        pp.extend(p)
        nn.extend(n)
    return aa, pp, nn


def generate_triplets(D,
                      limit=64,
                      speaker=None,
                      positive_strategy='Random',
                      negativesemihard=True,
                      negativehard=True):
    """
    Generate set of triplets per each centroid [C,P,N] (centroid vs anchor)
    with requirement of selecting M negative examples per each class.
    """
    if speaker is None:
        raise Exception('Speaker not provided')
    D.reset()
    positive_pool = D.get_siblings(speaker=speaker, limit=300, flag=False)
    if len(positive_pool) == 0:
        return [], [], []
    np.random.shuffle(positive_pool)

    aa, pp, nn = [], [], []
    while True:
        for anchor in positive_pool:
            negative_pool = D.get_opposites(speaker=speaker,
                                            limit=300,
                                            flag=False)
            if negative_pool is None or len(negative_pool) == 0:
                return [], [], []
            a, p, n = sub_generate_triplet_pair(
                D=D,
                anchor=anchor,
                positive_pool=positive_pool,
                negative_pool=negative_pool,
                margin=.3,
                positive_strategy=positive_strategy,
                negativesemihard=negativesemihard,
                negativehard=negativehard)
            if a is not None:
                aa.append(a)
                pp.append(p)
                nn.append(n)
            if len(aa) >= limit:
                return aa, pp, nn
        else:
            break
    return aa, pp, nn


def train_triplet_model(D, model, params, optimizer, anchors, positives,
                        negatives, epoch, criterion):
    device = next(model.parameters()).device
    # Prepare data for dataloader: embedding for anchor, spectrograms for positive and negative
    aa = [el['embedding'] for el in anchors]
    pp = [D.cache_read(el['cacheId']) for el in positives]
    #pp = [D.cache_read(el['cacheId']) for el in positives]
    nn = [D.cache_read(el['cacheId']) for el in negatives]

    trainingSet = TripletDataset(aa, pp, nn)
    trainingGenerator = torch.utils.data.DataLoader(trainingSet, **params)
    losses = []
    # can't move optimizer inside batch calc bcs we need to recalculate
    # all embeddings and calculate new centroids for each pass
    # Also that is the reason lr is so low, since gradient accumulation after
    # each backward pass
    optimizer.zero_grad()
    for ba, bp, bn in trainingGenerator:
        if len(ba) == 1:
            continue  # if we have 1 triplet pair batchnorm will fail
        mp = model(bp.float().to(device))
        mn = model(bn.float().to(device))
        loss = criterion(ba.to(device), mp, mn, epoch)
        loss.backward()
        losses.append(loss.item())
    optimizer.step()
    return losses
