import numpy as np
from scipy.spatial.distance import cdist
import libs.functions as _fn


def top1(candidates, centroids, truths):
    centroidVals = [np.squeeze(el[1]) for el in centroids]
    distMatrix = cdist(np.asarray(candidates),
                       np.asarray(centroidVals),
                       metric='cosine')
    bestIndex = np.argmin(distMatrix, axis=1)
    predictions = [centroids[el][0] for el in bestIndex]
    return np.sum([True for a, b in zip(predictions, truths) if a == b
                   ]) / len(predictions)


def top5(candidates, centroids, truths):
    centroidVals = [np.squeeze(el[1]) for el in centroids]
    distMatrix = cdist(np.asarray(candidates),
                       np.asarray(centroidVals),
                       metric='cosine')
    bestIndices = np.argpartition(distMatrix, kth=5, axis=1)[:, :5]
    predictions = [[centroids[el][0] for el in rec] for rec in bestIndices]
    sum = np.sum([True for a, b in zip(predictions, truths) if b in a])
    return sum / len(predictions)


def top5probabilities(candidates, centroids):
    centroidValues = [np.squeeze(el[1]) for el in centroids]
    candidatesValues = [el['embedding'] for el in candidates]
    distMatrix = cdist(np.asarray(candidatesValues),
                       np.asarray(centroidValues),
                       metric='cosine')
    bestIndices = np.argpartition(distMatrix, kth=5, axis=1)[:, :5]
    predictions = [[centroids[el][0] for el in rec] for rec in bestIndices]
    distances = np.asarray([[distMatrix[i][j] for j in el]
                            for i, el in enumerate(bestIndices)])
    probabilities = [[
        distances[i][j] / np.sum(distances[i])
        for j in range(distances[i].shape[0])
    ] for i in range(distances.shape[0])]
    output = [[{
        predictions[i][j]: probabilities[i][j]
    } for j in range(len(predictions[i]))] for i in range(len(predictions))]
    return output


def sequential_probability(candidates, centroids, truth):
    probs_all = top5probabilities(candidates, centroids)
    speakers = {}
    for probs_step in probs_all:
        for probs in probs_step:
            for speaker, prob in probs.items():
                if speaker in speakers.keys():
                    speakers[speaker] += prob
                else:
                    speakers[speaker] = prob
    output = dict(
        sorted(speakers.items(), key=lambda item: item[1], reverse=True)[:5])
    #output = [{key: value/sum(output.values())} for (key,value) in output.items()]
    if truth in output.keys():
        if truth == list(output.keys())[0]:
            return (True, True)
        else:
            return (False, True)
    else:
        return (False, False)


def test(model, D, T):
    _fn.report("-------------- Testing ----------------")
    # retrieve non-augmented records for dataset
    _fn.report("Updating embeddings")
    # Test on smaller subsets (64 speakers), but every 10 epoch test on full set

    D.update_embeddings(model)
    T.update_embeddings(model)

    _fn.report("Calculating centroids")
    centroids = D.calculate_centroids()
    # get N random training records and compare with centroids

    D.reset()
    testCandidates = D.get_random_records(limit_per_speaker=10, flag=False)
    top1train = np.round(
        100 *
        top1(candidates=[np.squeeze(el['embedding']) for el in testCandidates],
             centroids=centroids,
             truths=[el['speaker'] for el in testCandidates]),
        decimals=2)
    top5train = np.round(
        100 *
        top5(candidates=[np.squeeze(el['embedding']) for el in testCandidates],
             centroids=centroids,
             truths=[el['speaker'] for el in testCandidates]),
        decimals=2)

    _fn.report(f'Top 1 training accuracy dist(sample, centroids): {top1train}')
    _fn.report(f'Top 5 training accuracy dist(sample, centroids): {top5train}')

    T.reset()
    testCandidates = T.get_random_records(limit_per_speaker=10, flag=False)
    top1test = np.round(
        100 *
        top1(candidates=[np.squeeze(el['embedding']) for el in testCandidates],
             centroids=centroids,
             truths=[el['speaker'] for el in testCandidates]),
        decimals=2)
    top5test = np.round(
        100 *
        top5(candidates=[np.squeeze(el['embedding']) for el in testCandidates],
             centroids=centroids,
             truths=[el['speaker'] for el in testCandidates]),
        decimals=2)
    _fn.report(f'Top 1 test accuracy dist(sample, centroids): {top1test}')
    _fn.report(f'Top 5 test accuracy dist(sample, centroids): {top5test}')
    return top1train, top5train, top1test, top5test
