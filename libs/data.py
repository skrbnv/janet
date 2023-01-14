import numpy as np
import libs.functions as _fn
#import libs.visualization as _viz
import random
import torch
import pickle
from scipy.spatial.distance import cdist
import os
import itertools
import gc
from torch.utils.data import Dataset as UtilsDataset
import shutil
import tqdm


def cache_write(id, arr, cache=None):
    if cache is None:
        raise IOError('Missing cache path, cannot read data')
    if not os.path.isdir(cache):
        input('Cache directory doesn\'t exist. Press any key to create >> ')
        _fn.mkdir17(cache)
    storages = [f.path for f in os.scandir(cache) if f.is_dir()]
    if len(storages) == 0:
        sp = os.path.join(cache, f'{(1):010d}')
        _fn.mkdir17(sp)
        storages = [sp]
    latest_storage = max(storages, key=os.path.basename)
    count = len(os.listdir(latest_storage))
    if count < 10000:
        current_storage = latest_storage
    else:
        current_storage = os.path.join(cache, f'{(len(storages)+1):010d}')
        _fn.mkdir17(current_storage)
    try:
        with open(os.path.join(current_storage, id), 'wb') as f:
            np.save(f, arr)
    except Exception as e:
        raise Exception(
            f'Error while trying to save spectrogram {id} to {current_storage}, {e.args}'
        )
    return True


class FileStorageIndex():
    def __init__(self, paths) -> None:
        self.paths = paths
        self.storages = None
        self.index = {}
        self.rescan()

    def rescan(self):
        self.storages = []
        for path in self.paths:
            self.storages.extend(
                [f.path for f in os.scandir(path) if f.is_dir()])
        for storage in tqdm.tqdm(self.storages,
                                 desc="Rebuilding cache indices"):
            for f in os.listdir(storage):
                if not os.path.isdir(os.path.join(storage, f)):
                    self.index[f] = os.path.join(storage, f)

    def where(self, key):
        if key in self.index.keys():
            return self.index[key]
        else:
            return None

    def filter(self, keys):
        dks = []
        for key in self.index.keys():
            if key not in keys:
                dks.append(key)
            else:
                keys.remove(key)
        for dk in dks:
            del self.index[key]

    def __len__(self):
        return len(self.index)

    def empty(self):
        self.index = {}


class Dataset(UtilsDataset):
    def __init__(self,
                 cache_paths,
                 data=None,
                 useindex=False,
                 filename=None,
                 caching=False,
                 force_even=False):
        if data is not None and filename is not None:
            raise Exception(
                'Cannot create dataset with both data and filename passed as parameters'
            )

        self.cache_paths = [cache_paths] if isinstance(cache_paths,
                                                       str) else cache_paths
        self.report = print
        self.dm = None
        self.data = {}
        self.caching = caching
        self.cache = {}
        self.speakers = None
        self.index = FileStorageIndex(
            cache_paths) if useindex is True else None
        if data is not None:
            self.data = data
            if force_even:
                self.make_even()
            self.speakers = self.get_unique_speakers()
        elif filename is not None:
            _fn.report(f'Loading dataset from {os.path.basename(filename)}')
            self.load(filename)
            self.make_even()
            self.speakers = self.get_unique_speakers()

    def __getitem__(self, index, process_label=1):
        '''
        process_label = 0: Do not process
                      = 1: One Hot encoding
                      = 2: index encoding
        '''
        input = self.cache_read(self.data[index]['cacheId'])

        if process_label == 1:
            label = self.encode_speaker_one_hot(self.data[index]['speaker'])
        elif process_label == 2:
            label = self.encode_speaker_index(self.data[index]['speaker'])
        else:
            label = self.data[index]['speaker']
        info = self.data[index].copy()
        del info['cacheId']
        del info['selected']
        del info['embedding']
        del info['_id']
        del info['segments']
        return input, label, info

    def make_even(self):
        if len(self.data) % 2 == 1:
            self.data.popitem()
            return True
        else:
            return False

    def load(self, filename):
        with open(filename, 'rb') as handle:
            self.data = pickle.load(handle)
        return True

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
                    f'Dataset file ({filename}) already exists. Do you want to overwrite it? (Y/n)'
                ).lower()
                if response == 'y':
                    break
                elif response == 'n':
                    return False

        with open(filename, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def append(self, record):
        pos = self.__len__()
        self.data[pos] = record
        self.data[pos]['_id'] = pos
        self.speakers = None
        return True

    def encode_speaker_one_hot(self, speaker):
        speakers = self.get_unique_speakers()
        output = np.zeros(len(speakers), dtype=np.float32)
        try:
            spk_idx = speakers.index(speaker)
        except Exception:
            return None
        output[spk_idx] = 1
        return output

    def encode_speaker_index(self, speaker):
        speakers = self.get_unique_speakers()
        try:
            spk_idx = speakers.index(speaker)
            return spk_idx
        except Exception:
            return None

    def cache_write(self, id, obj):
        return cache_write(id, obj, cache=random.choice(self.cache_paths))

    def cache_read_wrapper(self, index):
        return self.cache_read(self.data[index]['cacheId'])

    def get_location(self, id):
        for cache_path in self.cache_paths:
            storages = [f.path for f in os.scandir(cache_path) if f.is_dir()]
            for storage in storages:
                if os.path.isfile(os.path.join(storage, id)):
                    return storage

    def subread(self, id):
        location = self.index.where(
            id) if self.index is not None else os.path.join(
                self.get_location(id), id)
        with open(location, 'rb') as f:
            try:
                arr = np.load(f)
            except Exception as e:
                raise IOError(
                    f'Encountered exception when trying to read file {id}, exception message: {e}'
                )
        return arr

    def cache_read(self, id):
        if self.caching is True:
            if id in self.cache.keys():
                return self.cache[id]
            else:
                arr = self.subread(id)
                self.cache[id] = arr
                return arr
        else:
            return self.subread(id)
        # np.squeeze(arr, axis=0)  # .T

    def trimmed(self, batch_size, procs=2):
        '''
        We need to remove last n records from dataset to fit it following
        requirements:
        D%3 == 0 (divided by 3),
        D%batch_size != 1 (does not have 1 remaining element for batchnorm)
        (D/3)%batch_size != 1 (same but for 1/3 for embeddings part)
        '''
        D_len = self.__len__()
        for i in range(D_len):
            if (D_len - i) % procs > 0:
                continue
            if (D_len - i) % batch_size == 1:
                continue
            if ((D_len - i) / procs) % batch_size == 1:
                continue
            return dict(itertools.islice(self.data.items(), D_len - i))

    def get_unique_speakers(self):
        if self.speakers is not None:
            return self.speakers
        else:
            speakers = set(el['speaker'] for el in self.data.values())
            return sorted(list(speakers))

    def get_speakers_subset(self, max_speakers=None, shuffle=True):
        speakers = self.get_unique_speakers()
        if shuffle is True:
            random.shuffle(speakers)
        return speakers[:max_speakers]

    def available(self):
        return sum(el['selected'] is False for el in self.data.values())

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
            embeddings = [
                el['embedding'] for el in self.data.values()
                if el['speaker'] == speaker and el['embedding'] is not None
            ]
            centroids.append([speaker, np.mean(embeddings, axis=0)])
        if s is not None:
            return centroids[0]
        else:
            return centroids

    def distance(self, id1, id2):
        assert self.dm is not None, 'Distance matrix was not calculated before'
        return self.dm[id1, id2]

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

    def update_embeddings(self, model, batch_size=32):
        device = next(model.parameters()).device
        model.eval()
        stop_value = self.__len__() - 1
        keys = []
        spectrograms = []
        for i, key in enumerate(self.data.keys()):
            keys.append(key)
            spg = self.cache_read(self.data[key]['cacheId'])
            spectrograms.append(spg)
            if len(keys) == batch_size or i == stop_value:
                with torch.no_grad():
                    embeddings = model(
                        torch.FloatTensor(np.array(spectrograms)).to(device))
                for j, key in enumerate(keys):
                    self.data[key]['embedding'] = embeddings[j].detach().cpu(
                    ).numpy()
                keys = []
                spectrograms = []
            if i % 10000 == 0:
                gc.collect()
        model.train()
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
            self.report("No unflagged records left")
            return None

        length = self.__len__()
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
                siblings = self.get_siblings(speaker,
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
            distCandidate = _fn.distance(embedding, record['embedding'])
            if distCandidate < minDistance:
                minDistance = distCandidate
                bestChoice = record
        if flag:
            self.flag_selected(bestChoice['_id'])
        return bestChoice

    def get_siblings(self, speaker, limit=-1, flag=True):
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

    def get_farthest_sibling(self, id):
        speaker = self.data[id]['speaker']
        siblings = self.get_siblings(speaker, flag=False)
        if len(siblings) == 0:
            raise Exception(f'No siblings found for speaker {speaker}')
        d = 0
        for sibling in siblings:
            candidate = self.distance(id, sibling['_id'])
            if candidate > d:
                d = candidate
                best_id = sibling['_id']
        return self.data[best_id]

    def get_closest_opposite(self, id, limit_scope=300):
        speaker = self.data[id]['speaker']
        opposites = self.get_opposites(speaker, limit=limit_scope, flag=False)
        d = np.finfo(np.float32).max
        for opposite in opposites:
            candidate = self.distance(id, opposite['_id'])
            if candidate < d:
                d = candidate
                best_id = opposite['_id']
        return self.data[best_id]

    def get_opposites(self, speaker, limit=1, flag=True):
        """ Returns random records not belonging to same speaker
        """

        length = self.__len__()
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
                                                max_records,
                                                speakers_filter,
                                                augmentations_filter=[],
                                                useindex=False,
                                                caching=False):
        # adding sorted just in case order follows initial order during creation,
        # when same speaker records were added sequentially
        repaug = f' with augmentations: {augmentations_filter}' if len(
            augmentations_filter) > 0 else ''
        _fn.report(
            f'Generating randomized subset with {len(speakers_filter)} speakers, up to {max_records} records each'
            + repaug)
        current_speaker = None
        records = []
        output = {}
        counter = 0
        stopval = self.__len__() - 1
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
        return Dataset(data=output,
                       cache_paths=self.cache_paths,
                       useindex=useindex,
                       caching=caching)

    def get_randomized_subset(self,
                              max_records,
                              speakers_filter,
                              useindex=False):
        return self.get_randomized_subset_with_augmentation(
            speakers_filter=speakers_filter,
            augmentations_filter=[],
            max_records=max_records,
            useindex=useindex)

    def augment(self, augmentations=[], extras={}):
        # ignore label smoothing (!?)

        # cache data

        # generate and cache augmented data
        return True
        '''
        Dsub.augment(augmentations=CFG.AUGMENTATIONS,
                     extras={'ambient': ambient})
        '''

    def visualize(self, epoch, samples=0):
        speakers = self.get_unique_speakers()
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

    def calculate_distances(self, metric='cosine'):
        # since all subsets are enumerated on creation, we will not create hash table, it is too slow
        k = list(self.data.keys())
        e = [self.data[key]['embedding'] for key in k]
        self.dm = cdist(e, e, metric=metric)
        return self.dm

    def get_augmentations_list(self):
        augms = list(set([el['augm'] for el in self.data]))
        output = []
        for augm in augms:
            if augm == 1:
                augm_name = 'white noise'
            elif augm == 2:
                augm_name = 'music overlay'
            elif augm == 3:
                augm_name = 'ambient overlay'
            else:
                continue
            output.append(augm_name)
        return output

    def __len__(self):
        return len(self.data)

    count = __len__
    length = __len__
    total = __len__


def cache_validate(id, cache):
    storages = [f.path for f in os.scandir(cache) if f.is_dir()]
    for storage in storages:
        if os.path.isfile(os.path.join(storage, id)):
            return True
    else:
        return False


def merge_cache(primary, relocated):
    relocated_storages = [d.path for d in os.scandir(relocated) if d.is_dir()]
    for i, relocated_storage in enumerate(relocated_storages):
        print(f'Merging caches, batch {i+1} out of {len(relocated_storages)}')
        filelist = os.listdir(relocated_storage)
        for file in filelist:
            primary_storages = [
                d.path for d in os.scandir(primary) if d.is_dir()
            ]
            latest_primary_storage = max(primary_storages,
                                         key=os.path.basename)
            count = len(os.listdir(latest_primary_storage))
            if count < 10000:
                target_storage = latest_primary_storage
            else:
                target_storage = os.path.join(
                    primary, f'{(len(primary_storages)+1):010d}')
                try:
                    os.mkdir(target_storage)
                except Exception as e:
                    raise Exception(
                        f'Cannot create directory {target_storage} {e.args}')
            shutil.move(os.path.join(relocated_storage, file),
                        os.path.join(target_storage, file))
    #input("Press enter to delete folders in secondary cache")
    for rf in relocated_storages:
        shutil.rmtree(rf)


def cache_summary(cache):
    storages = [d.path for d in os.scandir(cache) if d.is_dir()]
    total = sum([len(os.listdir(st)) for st in storages])
    return total


def save_records(speaker, records, D):
    ''' Shortcut for saving list of records into Dataset '''
    if records is not None:
        for record in records:
            D.append({
                'speaker': speaker,
                'sample': record['sample'],
                'position': record['position'],
                'cacheId': record['cacheId'],
                'augmentation': record['augmentation'],
                'segments': record['segments'],
                'rms': record['rms'],
                'selected': False,
                'embedding': None
            })
    else:
        print(f'No records returned for speaker {speaker}, skipping')
