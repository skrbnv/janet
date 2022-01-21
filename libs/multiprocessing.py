import os
import libs.functions as _fn
import random
import numpy as np
from libs.data import cache_write
import pickle
import cv2


def set_affinity_on_worker():
    """When a new worker process is created, the affinity is set to all CPUs"""
    os.system("taskset -p 0xffffff %d" % os.getpid())


def mp_worker(id, folder, length, config):
    ''' Multiprocessing worker, reads list of audio files per speaker,
        generating records and spectrograms and saves them '''

    speaker = os.path.basename(folder)
    _fn.report(
        f'Processing speaker {speaker}: {id} out of {length} (PID: {os.getpid()})'
    )
    records = []
    samples = _fn.raw_audio_by_dir(folder, config['MAX_SAMPLES_PER_SPEAKER'])

    for sample in samples:
        rcs = process_sample(sample, config)
        if rcs is not None:
            records.extend(rcs)

    if config['AUXILLARY_CACHE']:
        cache_path = random.choice(
            [config['PRIMARY_CACHE'], config['AUXILLARY_CACHE']])
    else:
        cache_path = config['PRIMARY_CACHE']
    if records is not None:
        for record in records:
            try:
                cache_write(record['cacheId'], record['spectrogram'],
                            cache_path)
            except Exception as e:
                raise Exception(
                    'Exception raised while trying to save spectrogram',
                    record['cacheId'], np.sum(record['spectrogram']), e.args)
        for record in records:
            del record['spectrogram']
    if config['RECORDS_DUMP']:
        with open(os.path.join(config['RECORDS_DUMP'], speaker), 'wb') as f:
            pickle.dump((speaker, records),
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return len(records)
    return speaker, records


def process_sample(sample, config):
    ''' Function responsible for processing single sample into set of spectrograms '''
    #audioEmpzd = _fn.preemphasize(sample[1])
    audioEmpzd = sample[1]
    augmentations = [0]
    if config['WHITENOISE']:
        augmentations.append(1)
    records = []
    for augmentation in augmentations:
        spectrograms, rms = _fn.generate_slices(
            audioEmpzd,
            length=config['SLICE_MS'] / 1000,
            sr=16000,
            step=config['STEP_MS'] / 1000,
            trim=config['TRIM_MS'] / 1000,
            strategy=config['SLICING_STRATEGY'],
            min_len=config['SKIPSHORTSLICES'],
            n_mels=config['MEL_BANKS'],
            augm=augmentation,
            config=config)

        if spectrograms is None:
            return None

        #shuffle slices to be able to pick spectrograms within sample in random order
        indices = np.arange(len(spectrograms))
        if config['PICK_RANDOM_SPECTROGRAMS']:
            np.random.shuffle(indices)
            spectrograms = [spectrograms[i] for i in indices]
        inner_counter = 0
        for (spectrogram, augm, segm), eachindex in zip(spectrograms, indices):
            if inner_counter == config['MAX_SPECTROGRAMS_PER_SAMPLE']:
                break
            spectrogram += 40
            spectrogram /= 40
            if config['FORCE_SHAPE']:
                spectrogram_resized = cv2.resize(
                    spectrogram, dsize=config['FORCE_SHAPE_SIZE'])[np.newaxis]
            else:
                spectrogram_resized = spectrogram[np.newaxis].astype(
                    np.float32)

            cache_id = f'{sample[0]}_{augm}-{eachindex.item()}'
            records.append({
                'sample': sample[0],
                'position': eachindex.item(),
                'cacheId': cache_id,
                'spectrogram': spectrogram_resized,
                'augmentation': augm,
                'segments': segm,
                'rms': rms
            })
            inner_counter += 1
    return records if len(records) > 0 else None
