import numpy as np
from libs.functions import report, distance
import libs.visualization as _viz
import random
import torch
import pickle
from scipy.spatial.distance import cdist
import os
import itertools
import gc
import psutil


class Dataset:
    def __init__(self, data=None):
        if data is None:
            self.data = {}
        else:
            self.data = data
        self.cache_path = None
        self.report = print

    def load(self, filename):
        with open(filename, 'rb') as handle:
            self.data = pickle.load(handle)
        return self.data

    def save(self, filename):
        if filename is None:
            self.report(
                'No filename for saving dataset was provided. Saving with "default.dt" name to "datasets" subfolder'
            )
            filename = "./datasets/dataset.dt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.isfile(filename):
            while True:
                response = input(
                    "File already exists. Do you want to overwrite it? (Y/n)"
                ).lower()
                if response == 'y':
                    break
                elif response == 'n':
                    return False

        with open(filename, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def cache_write(self, id, arr, cache):
        if cache is None:
            if self.cache_path is None:
                raise IOError('Missing cache path, cannot read data')
            else:
                cache = self.cache_path
        if not os.path.isdir(cache):
            input(
                'Cache directory doesn\'t exist. Press any key to create >> ')
            os.mkdir(cache)
        storages = [f.path for f in os.scandir(cache) if f.is_dir()]
        if len(storages) == 0:
            sp = os.path.join(cache, f'{(1):010d}')
            os.mkdir(sp)
            storages = [sp]
        latest_storage = max(storages, key=os.path.basename)
        count = len(os.listdir(latest_storage))
        if count < 10000:
            current_storage = latest_storage
        else:
            current_storage = os.path.join(cache, f'{(len(storages)+1):010d}')
            os.mkdir(current_storage)
        with open(os.path.join(current_storage, id), 'wb') as f:
            np.save(f, arr)
        return True

    def cache_read(self, id, cache=None):
        if cache is None:
            if self.cache_path is None:
                raise IOError('Missing cache path, cannot read data')
            else:
                cache = self.cache_path
        storages = [f.path for f in os.scandir(cache) if f.is_dir()]
        for storage in storages:
            if os.path.isfile(os.path.join(storage, id)):
                with open(os.path.join(storage, id), 'rb') as f:
                    try:
                        arr = np.load(f)
                    except Exception as e:
                        raise IOError(
                            f'Encountered exception when trying to read file {os.path.join(storage, id)}, exception message: {e}'
                        )
                return np.squeeze(arr, axis=0)  # .T
        raise Exception("No such file")

    def trimmed(self, batch_size=None):
        '''
        We need to remove last n records from dataset to fit it following
        requirements:
        D%3 == 0 (divided by 3),
        D%batch_size != 1 (does not have 1 remaining element for batchnorm)
        (D/3)%batch_size != 1 (same but for 1/3 for embeddings part)
        '''
        if batch_size is None:
            raise Exception("Trim procedure missing batch size")
        D_len = len(self.data)
        for i in range(len(self.data)):
            if (D_len - i) % 3 > 0:
                continue
            if (D_len - i) % batch_size == 1:
                continue
            if ((D_len - i) / 3) % batch_size == 1:
                continue
            return dict(itertools.islice(self.data.items(), D_len - i))

    def get_unique_speakers(self):
        speakers = set(el['speaker'] for el in self.data.values())
        return list(speakers)

    def get_speakers_subset(self, max_speakers=None, shuffle=True):
        if max_speakers is None:
            raise Exception("Number of speakers not provided")
        speakers = self.get_unique_speakers()
        if shuffle is True:
            random.shuffle(speakers)
        return speakers[:max_speakers]

    def available(self):
        return sum(el['selected'] is False for el in self.data.values())

    def count(self):
        return len(self.data)

    total = count

    def flag_selected(self, id):
        if self.data[id]['selected'] is True:
            return False
        else:
            self.data[id]['selected'] = True
            return True

    flag = flag_selected
    mark = flag_selected

    def reset(self):
        for el in self.data.values():
            update = {'selected': False}
            el.update(update)
        return True

    def overview(self):
        self.report("Overview of records' database:")
        speakers = self.get_unique_speakers()
        counters = {'speakers': len(speakers), 'samples': 0, 'records': 0}
        for speaker in speakers:
            records = [
                el['sample'] for el in self.data.values()
                if el['speaker'] == speaker
            ]
            samples = list(set(records))
            counters['samples'] += len(samples)
            counters['records'] += len(records)
        self.report("Speakers:", counters['speakers'], "Samples:",
                    counters['samples'], "Records:", counters['records'])

    def replace_report(self, fn):
        self.report = fn

    def calculate_centroids(self, s=None):
        speakers = self.get_unique_speakers()
        centroids = []
        for speaker in speakers:
            if s is not None and s != speaker:
                continue
            embeddings = []
            records = [
                el for el in self.data.values() if el['speaker'] == speaker
            ]
            for record in records:
                if record['embedding'] is not None:
                    embeddings.append(record['embedding'])
            centroids.append([speaker, np.mean(embeddings, axis=0)])
        if s is not None:
            return centroids[0]
        else:
            return centroids

    def calculate_centroids_dict(self, s=None):
        speakers = self.get_unique_speakers()
        centroids = {}
        for speaker in speakers:
            if s is not None and s != speaker:
                continue
            embeddings = []
            records = [
                el for el in self.data.values() if el['speaker'] == speaker
            ]
            for record in records:
                if record['embedding'] is not None:
                    embeddings.append(record['embedding'])
            centroids[speaker] = np.mean(embeddings, axis=0)
        if s is not None:
            return {s: centroids[0]}
        else:
            return centroids

    def update_embeddings(self, model, device, batch_size=32):
        stop_value = len(self.data) - 1
        keys = []
        spectrograms = []
        for i, (key, value) in enumerate(self.data.items()):
            keys.append(key)
            spectrograms.append(cacheRead(value['cacheId']))
            if len(keys) == batch_size or i == stop_value:
                with torch.no_grad():
                    embeddings = model(
                        torch.FloatTensor(spectrograms).to(device))
                for j, key in enumerate(keys):
                    self.data[key]['embedding'] = embeddings[j].detach().cpu(
                    ).numpy()
                keys = []
                spectrograms = []
            if i % 10000 == 0:
                gc.collect()
        return True

    def get_records_by_sample(self, sample):
        records = [el for el in self.data.values() if el['sample'] == sample]
        return records

    def get_samples_by_speaker(self, speaker):
        samples = set(el['sample'] for el in self.data.values()
                      if el['speaker'] == speaker)
        return list(samples)

    def speakers_iterator(self, batch_size=1, shuffle=True):
        """Yield successive batch-sized chunks from speakers list."""
        speakers = self.get_unique_speakers()
        if shuffle is True:
            random.shuffle(speakers)
        for i in range(0, len(speakers), batch_size):
            yield speakers[i:i + batch_size]

    def get_random_record(self, flag=True):
        """Return random non-flagged record"""
        if self.available() == 0:
            report("No unflagged records left")
            return None

        length = len(self.data)
        while True:
            index = random.randint(0, length - 1)
            if self.data[index]['selected'] is False:
                if flag:
                    self.data[index]['selected'] = True
                return self.data[index]

    def get_random_records(self,
                           amount=None,
                           limit_per_speaker=None,
                           flag=True):
        records = []

        if limit_per_speaker is not None and amount is None:
            speakers = self.get_unique_speakers()
            for speaker in speakers:
                siblings = self.getSiblings(speaker,
                                            limit=limit_per_speaker,
                                            flag=flag)
                records.extend(siblings)

        elif limit_per_speaker is None and amount is not None:
            for i in range(amount):
                record = self.get_random_record(flag=flag, fast=True)
                if record is not None:
                    records.append(record)
                else:
                    break
        else:
            raise Exception(
                'Cannot specify both amount and limit per speaker or not specify both'
            )
        return records

    def get_closest_record(self, speaker, embedding, flag=False):
        records = [el for el in self.data.values() if el['speaker'] == speaker]
        bestChoice = records[0]
        minDistance = np.finfo(np.float32).max
        for record in records:
            distCandidate = distance(embedding, record['embedding'])
            if distCandidate < minDistance:
                minDistance = distCandidate
                bestChoice = record
        if flag:
            self.flag_selected(bestChoice['_id'])
        return bestChoice

    def getSiblings(self, speaker, limit=-1, flag=True):
        """ Returns records belong to same speaker
        """
        records = [
            {
                key: value
            } for key, value in self.data.items()
            if value['speaker'] == speaker and value['selected'] is False
        ]
        if len(records) == 0:
            return []
        if limit > 0 and len(records) > limit:
            records = random.choices(records, k=limit)
        if flag:
            # get list of IDs
            ids = [list(record.keys())[0] for record in records]
            for id in ids:
                self.data[id]['selected'] = True
        return [list(record.values())[0] for record in records]

    def getOpposites(self, speaker, limit=1, flag=True):
        """ Returns random records not belonging to same speaker
        """

        length = len(self.data)
        # get number of opposites available for search but stop when we reach limit
        # to save computing time
        maxcount = 0
        for item in self.data.values():
            if item['selected'] is False and item['speaker'] != speaker:
                maxcount += 1
            if maxcount >= limit:
                break

        records = []
        while True:
            if len(records) >= maxcount:
                break
            index = random.randint(0, length - 1)
            if self.data[index]['selected'] is False and self.data[index][
                    'speaker'] != speaker:
                if flag:
                    self.data[index]['selected'] = True
                records.append(self.data[index])

        return records

    def get_randomized_subset_with_augmentation(self,
                                                speakers_filter=[],
                                                augmentations_filter=[],
                                                max_records=None):
        # adding sorted just in case order follows initial order during creation,
        # when same speaker records were added sequentially
        if max_records is None:
            raise Exception("Number of spectrograms per speaker not provided")
        current_speaker = None
        records = []
        output = {}
        counter = 0
        stopval = len(self.data.items()) - 1
        for index, (key, value) in enumerate(sorted(self.data.items())):
            # init
            if current_speaker is None:
                current_speaker = value['speaker']
            # index == stopval means we reached end of dictionary
            if current_speaker == value['speaker'] and index < stopval:
                if len(augmentations_filter
                       ) == 0 or value['augmentation'] in augmentations_filter:
                    records.append({'key': key, 'value': value})
            else:
                # if current_speaker changed to new or we reached end of dict
                if current_speaker in speakers_filter:
                    # check if current speaker is in 'approved' list
                    random.shuffle(records)
                    for record in records[:max_records:]:
                        output[counter] = record['value']
                        output[counter]['_id'] = counter
                        counter += 1
                # got to the record with next speaker
                records = [{'key': key, 'value': value}]
                current_speaker = value['speaker']
        return output

    def get_randomized_subset(self, speakers_filter=[], max_records=None):
        if max_records is None:
            raise Exception("Number of spectrograms per speaker not provided")
        return self.get_randomized_subset_with_augmentation(
            self,
            speakers_filter=speakers_filter,
            augmentations_filter=[],
            max_records=max_records)

    def get_hardest_speakers(centroids, limit=100):
        dm = cdist(np.asarray([c[1] for c in centroids]),
                   np.asarray([c[1] for c in centroids]),
                   metric='cosine')
        diag = (1 - np.tri(dm.shape[0], dm.shape[1], k=-1)) * np.tri(
            dm.shape[0], dm.shape[1], k=0) * 1e9
        dm = dm + diag
        min_idx = np.argsort(dm.flatten())
        idl = np.unravel_index(min_idx, (len(centroids), len(centroids)))
        cmb = []
        for i in range(len(idl[0])):
            if idl[0][i] not in cmb:
                cmb.append(idl[0][i])
            if idl[1][i] not in cmb:
                cmb.append(idl[1][i])
            if len(cmb) >= limit:
                cmb = cmb[:limit]
                break
        speakers = [centroids[i][0] for i in cmb]
        return speakers

    def visualize(self, epoch, samples=0):
        speakers = getUniqueSpeakers(self.data)
        dataset = []
        for speaker in speakers:
            subset = []
            searchResults = [
                el['embedding'] for el in self.data.values()
                if el['speaker'] == speaker
            ]
            if samples > 0:
                searchResults = random.choices(searchResults, k=samples)
                subset.extend(searchResults)
            dataset.append(subset)
        _viz.visualize(dataset=dataset, epoch=epoch)


def cacheWrite(id, arr, cache=None):
    if cache is None:
        raise Exception("Cache path not provided")
    if not os.path.isdir(cache):
        input('Cache directory doesn\'t exist. Press any key to create >> ')
        os.mkdir(cache)
    storages = [f.path for f in os.scandir(cache) if f.is_dir()]
    if len(storages) == 0:
        sp = os.path.join(cache, f'{(1):010d}')
        os.mkdir(sp)
        storages = [sp]
    latest_storage = max(storages, key=os.path.basename)
    count = len(os.listdir(latest_storage))
    if count < 10000:
        current_storage = latest_storage
    else:
        current_storage = os.path.join(cache, f'{(len(storages)+1):010d}')
        os.mkdir(current_storage)
    with open(os.path.join(current_storage, id), 'wb') as f:
        np.save(f, arr)
    return True


def cacheRead(id, cache=None):
    if cache is None:
        raise Exception("Cache path not provided")
    storages = [f.path for f in os.scandir(cache) if f.is_dir()]
    for storage in storages:
        if os.path.isfile(os.path.join(storage, id)):
            with open(os.path.join(storage, id), 'rb') as f:
                try:
                    arr = np.load(f)
                except Exception as e:
                    raise IOError(
                        f'Encountered exception when trying to read file {os.path.join(storage, id)}, exception text: {e}'
                    )
            return np.squeeze(arr, axis=0)  # .T
    raise Exception("No such file")


def writeDataset(D, filename=None):
    if filename is None:
        filename = "dataset.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)


def readDataset(filename):
    with open(filename, 'rb') as handle:
        D = pickle.load(handle)
    return D


def trimDataset(D, batch_size=None):
    '''
    We need to remove last n records from dataset to fit it following
    requirements:
        D%3 == 0 (divided by 3),
        D%BATCH_SIZE != 1 (does not have 1 remaining element for batchnorm)
        (D/3)%BATCH_SIZE != 1 (same but for 1/3 for embeddings part)
    '''
    if batch_size is None:
        raise Exception("Batch size not provided")
    D_len = len(D)
    for i in range(len(D)):
        if (D_len - i) % 3 > 0:
            continue
        if (D_len - i) % batch_size == 1:
            continue
        if ((D_len - i) / 3) % batch_size == 1:
            continue
        return dict(itertools.islice(D.items(), D_len - i))


def getSpeakersSubset(D, max_speakers=None, shuffle=True):
    if max_speakers is None:
        raise Exception("Number of speakers not provided")
    speakers = getUniqueSpeakers(D)
    if shuffle is True:
        random.shuffle(speakers)
    return speakers[:max_speakers]


def speakersIterator(D, batch_size=1, shuffle=True):
    """Yield successive batch-sized chunks from speakers list."""
    speakers = getUniqueSpeakers(D)
    if shuffle is True:
        random.shuffle(speakers)
    for i in range(0, len(speakers), batch_size):
        yield speakers[i:i + batch_size]


def getRandomizedSubsetWithAugmentation(D,
                                        speakers_filter=[],
                                        augmentations_filter=[],
                                        max_records=None):
    # adding sorted just in case order follows initial order during creation,
    # when same speaker records were added sequentially
    if max_records is None:
        raise Exception("Number of spectrograms per speaker not provided")
    current_speaker = None
    records = []
    output = {}
    counter = 0
    stopval = len(D.items()) - 1
    for index, (key, value) in enumerate(sorted(D.items())):
        # init
        if current_speaker is None:
            current_speaker = value['speaker']
        # index == stopval means we reached end of dictionary
        if current_speaker == value['speaker'] and index < stopval:
            if len(augmentations_filter
                   ) == 0 or value['augmentation'] in augmentations_filter:
                records.append({'key': key, 'value': value})
        else:
            # if current_speaker changed to new or we reached end of dict
            if current_speaker in speakers_filter:
                # check if current speaker is in 'approved' list
                random.shuffle(records)
                for record in records[:max_records:]:
                    output[counter] = record['value']
                    output[counter]['_id'] = counter
                    counter += 1
            # got to the record with next speaker
            records = [{'key': key, 'value': value}]
            current_speaker = value['speaker']
    reset(output)
    return output


def getRandomizedSubset(D, speakers_filter=[], max_records=None):
    if max_records is None:
        raise Exception("Number of spectrograms per speaker not provided")
    return getRandomizedSubsetWithAugmentation(D,
                                               speakers_filter=speakers_filter,
                                               augmentations_filter=[],
                                               max_records=max_records)


def reset(D):
    for el in D.values():
        update = {'selected': False}
        el.update(update)
    return True


def flag_selected(D, id):
    D[id]['selected'] = True
    return True


def recordsAvailable(D):
    return sum(el['selected'] is False for el in D.values())


def getUniqueSpeakers(D):
    speakers = set(el['speaker'] for el in D.values())
    return list(speakers)


def getSamples(D, speaker):
    samples = set(el['sample'] for el in D.values()
                  if el['speaker'] == speaker)
    return list(samples)


def getRecords(D, sample):
    records = [el for el in D.values() if el['sample'] == sample]
    return records


def overview(D):
    report("Overview of records' database:")
    speakers = getUniqueSpeakers(D)
    counters = {'speakers': len(speakers), 'samples': 0, 'records': 0}
    for speaker in speakers:
        records = [
            el['sample'] for el in D.values() if el['speaker'] == speaker
        ]
        samples = list(set(records))
        counters['samples'] += len(samples)
        counters['records'] += len(records)
    report("Speakers:", counters['speakers'], "Samples:", counters['samples'],
           "Records:", counters['records'])


def calculateCentroids(D, s=None):
    speakers = getUniqueSpeakers(D)
    centroids = []
    for speaker in speakers:
        if s is not None and s != speaker:
            continue
        embeddings = []
        records = [el for el in D.values() if el['speaker'] == speaker]
        for record in records:
            if record['embedding'] is not None:
                embeddings.append(record['embedding'])
        centroids.append([speaker, np.mean(embeddings, axis=0)])
    if s is not None:
        return centroids[0]
    else:
        return centroids


def calculateCentroidsAsDict(D, s=None):
    speakers = getUniqueSpeakers(D)
    centroids = {}
    for speaker in speakers:
        if s is not None and s != speaker:
            continue
        embeddings = []
        records = [el for el in D.values() if el['speaker'] == speaker]
        for record in records:
            if record['embedding'] is not None:
                embeddings.append(record['embedding'])
        centroids[speaker] = np.mean(embeddings, axis=0)
    if s is not None:
        return {s: centroids[0]}
    else:
        return centroids


def getHardestSpeakers(centroids, limit=100):
    dm = cdist(np.asarray([c[1] for c in centroids]),
               np.asarray([c[1] for c in centroids]),
               metric='cosine')
    diag = (1 - np.tri(dm.shape[0], dm.shape[1], k=-1)) * np.tri(
        dm.shape[0], dm.shape[1], k=0) * 1e9
    dm = dm + diag
    min_idx = np.argsort(dm.flatten())
    idl = np.unravel_index(min_idx, (len(centroids), len(centroids)))
    cmb = []
    for i in range(len(idl[0])):
        if idl[0][i] not in cmb:
            cmb.append(idl[0][i])
        if idl[1][i] not in cmb:
            cmb.append(idl[1][i])
        if len(cmb) >= limit:
            cmb = cmb[:limit]
            break
    speakers = [centroids[i][0] for i in cmb]
    return speakers


def updateEmbeddings(D, model, device, batch_size=32):
    stop_value = len(D) - 1
    keys = []
    spectrograms = []
    for i, (key, value) in enumerate(D.items()):
        keys.append(key)
        spectrograms.append(cacheRead(value['cacheId']))
        if len(keys) == batch_size or i == stop_value:
            with torch.no_grad():
                embeddings = model(torch.FloatTensor(spectrograms).to(device))
            for j, key in enumerate(keys):
                D[key]['embedding'] = embeddings[j].detach().cpu().numpy()
            keys = []
            spectrograms = []
        if i % 10000 == 0:
            gc.collect()
    return True


def getRandomRecord(D, flag=True, fast=False):
    if not fast and recordsAvailable(D) == 0:
        report("No unflagged records left")
        return None

    length = len(D)
    while True:
        index = random.randint(0, length - 1)
        if D[index]['selected'] is False:
            if flag:
                D[index]['selected'] = True
            return D[index]


def getRandomRecords(D, amount=None, limit_per_speaker=None, flag=True):
    records = []

    if limit_per_speaker is not None and amount is None:
        speakers = getUniqueSpeakers(D)
        for speaker in speakers:
            siblings = getSiblings(D,
                                   speaker,
                                   limit=limit_per_speaker,
                                   flag=flag)
            records.extend(siblings)

    elif limit_per_speaker is None and amount is not None:
        for i in range(amount):
            record = getRandomRecord(D, flag=flag, fast=True)
            if record is not None:
                records.append(record)
            else:
                break
    else:
        raise Exception(
            'Cannot specify both amount and limit per speaker or not specify both'
        )

    return records


def getClosestRecord(D, speaker, embedding, flag=False):
    records = [el for el in D.values() if el['speaker'] == speaker]
    bestChoice = records[0]
    minDistance = np.finfo(np.float32).max
    for record in records:
        distCandidate = distance(embedding, record['embedding'])
        if distCandidate < minDistance:
            minDistance = distCandidate
            bestChoice = record
    if flag:
        flag_selected(D, bestChoice['_id'])
    return bestChoice


# get record from the same speaker by speaker or by id (speaker has higher priority)
def getSiblings(D, speaker, limit=-1, flag=True):
    """ Returns records belong to same speaker
    """
    records = [{
        key: value
    } for key, value in D.items()
               if value['speaker'] == speaker and value['selected'] is False]
    if len(records) == 0:
        return []
    if limit > 0 and len(records) > limit:
        records = random.choices(records, k=limit)
    if flag:
        # get list of IDs
        ids = [list(record.keys())[0] for record in records]
        for id in ids:
            D[id]['selected'] = True
    return [list(record.values())[0] for record in records]


def getOpposites(D, speaker, limit=1, flag=True):
    """ Returns random records not belonging to same speaker
    """

    length = len(D)
    # get number of opposites available for search but stop when we reach limit
    # to save computing time
    maxcount = 0
    for item in D.values():
        if item['selected'] is False and item['speaker'] != speaker:
            maxcount += 1
        if maxcount >= limit:
            break

    records = []
    while True:
        if len(records) >= maxcount:
            break
        index = random.randint(0, length - 1)
        if D[index]['selected'] is False and D[index]['speaker'] != speaker:
            if flag:
                D[index]['selected'] = True
            records.append(D[index])

    return records


def visualize(D, epoch, samples=0):
    speakers = getUniqueSpeakers(D)
    dataset = []
    for speaker in speakers:
        subset = []
        searchResults = [
            el['embedding'] for el in D.values() if el['speaker'] == speaker
        ]
        if samples > 0:
            searchResults = random.choices(searchResults, k=samples)
            subset.extend(searchResults)
        dataset.append(subset)
    _viz.visualize(dataset=dataset, epoch=epoch)


# # # # # # # # #


def centroidDistances(centroids, filename=None):
    dm = cdist(np.asarray([c[1] for c in centroids]),
               np.asarray([c[1] for c in centroids]),
               metric='cosine')
    dm_masked = np.ma.masked_array(dm,
                                   np.tri(len(centroids), len(centroids), k=0),
                                   fill_value=np.nan)
    dm = dm_masked.flatten()
    if filename is not None:
        _viz.distances_histogram(dm[dm.mask is False], filename)
    return np.nanmin(dm), np.nanmean(dm), np.nanmax(dm)


def generateDistanceMatrixAndDump(centroids,
                                  filename="centroid_distances.csv"):
    distMatrix = cdist(np.asarray([c[1] for c in centroids]),
                       np.asarray([c[1] for c in centroids]),
                       metric='cosine')
    zeroed = distMatrix * (
        1 - np.tri(distMatrix.shape[0], distMatrix.shape[1], k=0))
    with open(filename, "w") as f:
        f.write(" ,")
        for c in centroids:
            f.write(f'{c[0]},')
        f.write('\n')
        for i in range(zeroed.shape[0]):
            f.write(f'{centroids[i][0]},')
            for j in range(zeroed.shape[1]):
                f.write(f'{zeroed[i,j]},')
            f.write('\n')
    return True
