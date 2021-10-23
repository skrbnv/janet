import numpy as np
import torch
#import libs.data as _db
import libs.functions as _fn
from libs.datasets import TripletDataset, CentroidDataset


def sub_generate_classic_triplet_pair(D,
                                      anchor,
                                      positive_pool,
                                      negative_pool,
                                      margin=.3,
                                      negativesemihard=True,
                                      negativehard=True):
    for positive in positive_pool:
        if positive['_id'] == anchor['_id']:
            continue
        dAP = _fn.distance(anchor['embedding'], positive['embedding'])

        np.random.shuffle(negative_pool)
        for negative in negative_pool:
            dAN = _fn.distance(anchor['embedding'], negative['embedding'])
            # semi hard selection: dAP < dAN < dAP+margin
            # hard selection: dAN < dAP
            if (negativesemihard and dAP < dAN
                    and dAN < dAP + margin) or (negativehard and dAP >= dAN):
                # _db.flag_selected(D, positive['_id'])
                # _db.flag_selected(D, negative['_id'])
                return (anchor, positive, negative)
    return None, None, None


def generate_classic_triplets(D, epoch, limit=64, speaker=None, centroid=None):
    """
    Generate set of triplets per each centroid [C,P,N] (centroid vs anchor)
    with requirement of selecting M negative examples per each class.
    """

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
            a, p, n = sub_generate_classic_triplet_pair(
                D, anchor, positive_pool, negative_pool, epoch)
            if a is not None:
                aa.append(a)
                pp.append(p)
                nn.append(n)
            if len(aa) >= limit:
                return aa, pp, nn
        else:
            break
    return aa, pp, nn


def train_triplet_model_dual(D, model, params, optimizer, anchors, positives,
                             negatives, epoch, device, criterion):
    # Prepare data for dataloader: embedding for anchor, spectrograms for positive and negative
    aa = [el['embedding'] for el in anchors]
    pp = [D.cache_read(el['cacheId']) for el in positives]
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
        mp, mn = model(bp.float().to(device), bn.float().to(device))
        loss = criterion(ba.to(device), mp, mn, epoch)
        loss.backward()
        losses.append(loss.item())
    optimizer.step()
    return losses


def generateTripletsViaCentroids(D,
                                 limit=64,
                                 speaker=None,
                                 centroid=None,
                                 positive_criterion='Random',
                                 margin=.3,
                                 negativesemihard=True,
                                 negativehard=True):
    """Generate set of triplets per each centroid [C,P,N] (centroid vs anchor)
    with requirement of selecting M negative examples per each class. 
    Idea here is to push negative examples for each and every class available
    from the centroid and pull positive examples closer.
    """

    anchors = []
    positives = []
    negatives = []

    D.reset()
    prototype = D.get_closest_record(speaker=speaker,
                                     embedding=centroid,
                                     flag=True)
    # get some number of positive examples to test

    while (len(anchors) < limit):
        positivePool = D.get_siblings(speaker=prototype['speaker'],
                                      limit=300,
                                      flag=False)
        np.random.shuffle(positivePool)
        # if positives retured none or error then skip this centroid
        if len(positivePool) == 0:
            break
        if positive_criterion == 'Random':
            while True:
                positive = np.random.choice(positivePool)
                if positive['_id'] != prototype['_id']:
                    break
        else:  # positive_criterion == 'Hard':
            maxDist = 0
            for posCandidate in positivePool:
                d = _fn.distance(np.asarray(prototype['embedding']),
                                 np.asarray(posCandidate['embedding']))
                if d < 0:
                    raise Exception('WTF distance < 0')
                if d >= maxDist:
                    maxDist = d
                    positive = posCandidate

        D.flag_selected(positive['_id'])
        dAP = _fn.distance(np.asarray(prototype['embedding']),
                           np.asarray(positive['embedding']))

        # get list of negative candidates
        negativePool = D.get_opposites(speaker=prototype['speaker'],
                                       limit=30,
                                       flag=False)
        np.random.shuffle(negativePool)
        # if no negative left exit the cycle
        if negativePool is None or len(negativePool) == 0:
            break
        # look for negative examples that fit criterion
        # until needed number is collected
        negativeFits = False
        for negCandidate in negativePool:
            dAN = _fn.distance(np.asarray(prototype['embedding']),
                               np.asarray(negCandidate['embedding']))
            # semi hard selection: dAP < dAN < dAP+margin
            if negativesemihard and dAP < dAN and dAN < (dAP + margin):
                #if dAN < dAP:
                negativeFits = True
            # hard selection: dAN < dAP
            elif negativehard and dAP >= dAN:
                negativeFits = True

            if negativeFits:
                D.flag_selected(negCandidate['_id'])
                anchors.append(D.cache_read(prototype['cacheId']))
                positives.append(D.cache_read(positive['cacheId']))
                negatives.append(D.cache_read(negCandidate['cacheId']))
                break

    if len(anchors) != len(positives) != len(negatives):
        raise Exception(
            "Error while generating triplets: number of anchors, positives and negatives do not match"
        )
    D.reset()
    return anchors, positives, negatives


def generateTripletsViaCentroidsOptimized(D,
                                          limit=64,
                                          speaker=None,
                                          centroid=None,
                                          positive_criterion='Random',
                                          margin=.3,
                                          negativesemihard=True,
                                          negativehard=True):
    """Generate set of triplets per each centroid [C,P,N] (centroid vs anchor)
    with requirement of selecting M negative examples per each class. 
    Idea here is to push negative examples for each and every class available
    from the centroid and pull positive examples closer.
    """
    if speaker is None:
        raise Exception('Speaker cannot be undefined')
    if centroid is None:
        raise Exception('Centroid cannot be undefined')
    if centroid is None:
        raise Exception('Epoch cannot be undefined')

    anchors = []
    positives = []
    negatives = []

    _db.reset(D)

    while (len(anchors) < limit):
        positivePool = _db.getSiblings(D,
                                       speaker=speaker,
                                       limit=300 if 300 > limit else limit,
                                       flag=False)
        np.random.shuffle(positivePool)
        # if positives retured none or error then skip this centroid
        if len(positivePool) == 0:
            break
        if positive_criterion == 'Random':
            positive = np.random.choice(positivePool)
        else:  # positive_criterion == 'Hard':
            maxDist = 0
            for posCandidate in positivePool:
                d = _fn.distance(centroid[1], posCandidate['embedding'])
                if d < 0:
                    raise Exception('WTF distance < 0')
                if d >= maxDist:
                    maxDist = d
                    positive = posCandidate

        _db.flag_selected(D, positive['_id'])
        dAP = _fn.distance(centroid[1], positive['embedding'])

        # get list of negative candidates
        negativePool = _db.getOpposites(D,
                                        speaker=speaker,
                                        limit=30,
                                        flag=False)
        # if no negative left exit the cycle
        if negativePool is None or len(negativePool) == 0:
            break
        # look for negative examples that fit criterion
        # until needed number is collected
        negativeFits = False
        for negCandidate in negativePool:
            dAN = _fn.distance(centroid[1], negCandidate['embedding'])
            # semi hard selection: dAP < dAN < dAP+margin
            if negativesemihard and dAP < dAN and dAN < (dAP + margin):
                #if dAN < dAP:
                negativeFits = True
            # hard selection: dAN < dAP
            elif negativehard and dAP >= dAN:
                negativeFits = True

            if negativeFits:
                _db.flag_selected(D, negCandidate['_id'])
                anchors.append(centroid[1])
                positives.append(_db.cacheRead(positive['cacheId']))
                negatives.append(_db.cacheRead(negCandidate['cacheId']))
                break

    if len(anchors) != len(positives) != len(negatives):
        raise Exception(
            "Error while generating triplets: number of anchors, positives and negatives do not match"
        )
    return anchors, positives, negatives


def trainCentroidModel(model, params, optimizer, centroids, siblings, epoch,
                       device, criterion):
    trainingSet = CentroidDataset(siblings)
    trainingGenerator = torch.utils.data.DataLoader(trainingSet, **params)
    losses = []
    # can't move optimizer inside batch calc bcs we need to recalculate
    # all embeddings and calculate new centroids for each pass
    # Also that is the reason lr is so low, since gradient accumulation after
    # each backward pass
    optimizer.zero_grad()
    for b_siblings in trainingGenerator:
        if len(b_siblings) == 1:
            continue  # if we have 1 triplet pair batchnorm will fail
        m_out = model(b_siblings['spg'].float().to(device))
        loss = criterion(centroids, m_out, b_siblings['spk'], device, epoch)
        loss.backward()
        losses.append(loss.item())
    optimizer.step()
    return losses
