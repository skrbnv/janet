import libs.functions as _fn
from libs.data import Dataset, cache_validate, merge_cache, save_records, cache_summary
import os
import pickle
from multiprocessing import Pool, cpu_count
import time
import argparse
import libs.multiprocessing as _mp
import shutil

if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        action='store_true',
                        default=False,
                        help='continue generating spectrograms')
    args = parser.parse_args()
    RESUME = args.resume

    # SETTINGS
    '''
    MEDIA_DIR = "/media/sergey/EXT4/datasets/TIMIT2/raw/sorted/test"
    DATASET_TARGET = "/media/sergey/EXT4/datasets/TIMIT2/simple/datasets/test.dt"
    PRIMARY_CACHE = "/media/sergey/EXT4/datasets/TIMIT2/simple/cache/test"
    AUXILLARY_CACHE = None
    RECORDS_DUMP = None
    MERGE_CACHES = False
    '''
    MEDIA_DIR = "/media/sergey/EXT4/datasets/TIMIT2/raw/sorted/validate"
    DATASET_TARGET = "/media/sergey/EXT4/datasets/TIMIT2/generated/simple/datasets/validate.dt"
    PRIMARY_CACHE = "/media/sergey/EXT4/datasets/TIMIT2/generated/simple/cache/validate"
    AUXILLARY_CACHE = None  # "/mnt/nvme1tb/datasets/voxceleb2/tiny/cache/test"
    RECORDS_DUMP = './records/tmp'
    MERGE_CACHES = False

    # AUXILLARY_CACHE is a temp cache on second disk drive to increase number of IO ops
    # RECORDS_DUMP, if exists, defines temp location to store records due to delays if
    # we sync adding records to single dataset

    FORCE_SHAPE = False
    FORCE_SHAPE_SIZE = (224, 224)
    SLICE_MS = 1920
    STEP_MS = 100
    MEL_BANKS = 64
    TRIM_MS = 100
    # number of milliseconds to trim
    # from the start and end of the
    # record. Problem of voxceleb dataset

    SLICING_STRATEGY = 'glue'
    # 'glue' non-silent parts into one then slice or
    # 'tokenize' - glue with tokens of removed silence inserted in between or
    # 'pad' each part and slice each one
    SKIPSHORTSLICES = .5
    # ignore (do not pad) slices with length less than some fraction of target

    SKIPFIRSTFRAME = False
    # Do not add first frame to set due to
    # a lot of examples ending up in outliers

    #SPECTROGRAMS#
    MAX_SPEAKERS = 5994
    MAX_SAMPLES_PER_SPEAKER = 10
    MAX_SPECTROGRAMS_PER_SAMPLE = 20
    PICK_RANDOM_SPECTROGRAMS = True
    TEST_RESULTS = False

    _fn.report(" ************************************************** ")
    _fn.report(" **            Spectrograms generation           ** ")
    _fn.report(" ************************************************** ")

    _fn.report("----------------- CURRENT CONFIG -----------------")
    _fn.report(f'Slicing strategy: {SLICING_STRATEGY}')
    _fn.report(f'Media dir: {MEDIA_DIR}')
    _fn.report(f'Dataset target: {DATASET_TARGET}')
    _fn.report(f'Cache dir: {PRIMARY_CACHE}')
    _fn.report(f'Auxilary cache dir: {AUXILLARY_CACHE}')
    _fn.report(f'Records temp dump dir: {RECORDS_DUMP}')
    _fn.report(f'Slice size, ms {SLICE_MS}')
    _fn.report(f'Step size, ms {STEP_MS}')
    _fn.report(f'Using up to {MAX_SAMPLES_PER_SPEAKER} samples per speaker')
    _fn.report(
        f'   with up to {MAX_SPECTROGRAMS_PER_SAMPLE} {"_randomly selected_" if PICK_RANDOM_SPECTROGRAMS else ""} spectrograms per sample'
    )
    _fn.report(f'   for up to {MAX_SPEAKERS} speakers,')
    _fn.report(
        f'Dumping records: {"True, to "+RECORDS_DUMP if RECORDS_DUMP else "False"}'
    )
    _fn.report(f'Melbanks: {MEL_BANKS}')
    _fn.report(f'Trimming audio, ms: {TRIM_MS}')
    _fn.report(f'Skipping first frame: {SKIPFIRSTFRAME}')
    input("Press any key to continue >> ")

    max_processes = cpu_count()
    step = max_processes * 10
    folders = [f.path for f in os.scandir(MEDIA_DIR) if f.is_dir()]
    if MAX_SPEAKERS > 0:
        folders = folders[:MAX_SPEAKERS]

    cache_paths = [PRIMARY_CACHE] if AUXILLARY_CACHE is None else [
        PRIMARY_CACHE, AUXILLARY_CACHE
    ]
    D = Dataset(cache_paths=cache_paths)
    #cache = _cw.CachedWrites('/media/sergey/3Tb1/cache')
    if RECORDS_DUMP:
        if not os.path.exists(RECORDS_DUMP):
            os.mkdir(RECORDS_DUMP)
    if RESUME:
        dump = _fn.filelist(RECORDS_DUMP)
        folders = [f for f in folders if os.path.basename(f) not in dump]
    limit = len(folders)
    config = {
        'MAX_SAMPLES_PER_SPEAKER': MAX_SAMPLES_PER_SPEAKER,
        'PRIMARY_CACHE': PRIMARY_CACHE,
        'AUXILLARY_CACHE': AUXILLARY_CACHE,
        'RECORDS_DUMP': RECORDS_DUMP,
        'SLICE_MS': SLICE_MS,
        'STEP_MS': STEP_MS,
        'TRIM_MS': TRIM_MS,
        'MAX_SPECTROGRAMS_PER_SAMPLE': MAX_SPECTROGRAMS_PER_SAMPLE,
        'SLICING_STRATEGY': SLICING_STRATEGY,
        'SKIPSHORTSLICES': SKIPSHORTSLICES,
        'MEL_BANKS': MEL_BANKS,
        'PICK_RANDOM_SPECTROGRAMS': PICK_RANDOM_SPECTROGRAMS,
        'FORCE_SHAPE': FORCE_SHAPE,
        'FORCE_SHAPE_SIZE': FORCE_SHAPE_SIZE,
    }
    if limit > 0:
        total_expected = 0
        pool = Pool(processes=max_processes,
                    initializer=_mp.set_affinity_on_worker)
        for i in range(0, limit, step):
            results = []
            for j in range(i, i + step):
                if j < limit:
                    result = pool.apply_async(
                        _mp.mp_worker, (j + 1, folders[j], limit, config))
                    results.append(result)
            if RECORDS_DUMP:
                #[result.wait() for result in results]
                while True:
                    time.sleep(1)
                    # catch exception is results are not ready yet
                    try:
                        ready = [result.ready() for result in results]
                        successful = [
                            result.successful() for result in results
                        ]
                    except Exception:
                        continue
                    if all(successful):
                        break
                    if all(ready) and not all(successful):
                        raise Exception(
                            f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}'
                        )
                if TEST_RESULTS:
                    num_generated = len(_fn.filelist(RECORDS_DUMP))
                    if not num_generated == min(i + step, limit):
                        raise Exception(
                            f'Mismatch in number of passed for processing speakers ({min(i + step, limit)}) and actually processed ({num_generated}) '
                        )
                    total_expected += sum(
                        [result._value for result in results])
                    total_generated = cache_summary(PRIMARY_CACHE)
                    if AUXILLARY_CACHE:
                        total_generated += cache_summary(AUXILLARY_CACHE)
                    if not total_expected == total_generated:
                        raise Exception(
                            f'Mismatch in number of reported spectrograms ({total_expected}) and actually generated ({total_generated}) '
                        )
            else:
                processed = 0
                while True:
                    if len(results) == 0:
                        break
                    for result in results:
                        if result.ready():
                            if not result.successful():
                                raise Exception(result._value)
                            speaker, records = result._value
                            print(f'Processing {speaker}')
                            save_records(speaker, records, D)
                            processed += 1
                            _fn.report(
                                f'In pool: {len(results)-1}, processed: {processed}'
                            )
                            results.remove(result)
                    else:
                        print('\u263d\u263d\u263d')
                        time.sleep(1)
                D.save(f'{DATASET_TARGET}-{i + step}')
                speakers = D.get_unique_speakers()
                with open('processed_speakers.pkl', 'wb') as f:
                    pickle.dump(speakers, f, protocol=pickle.HIGHEST_PROTOCOL)

        pool.close()
        pool.join()

    else:
        if not RESUME:
            raise IOError('No speakers\' dirs found in mentioned directory')
        #cache.finalize()

    if RECORDS_DUMP:
        dump = _fn.filelist(RECORDS_DUMP)
        processed = [os.path.join(RECORDS_DUMP, el) for el in dump]
        for i, el in enumerate(processed):
            with open(el, 'rb') as f:
                speaker, records = pickle.load(f)
                _fn.report(
                    f'{i} out of {len(processed)}: saving and validating speaker {speaker}'
                )
                save_records(speaker, records, D)
                # validate that cache file was created
                for record in records:
                    if cache_validate(record['cacheId'], PRIMARY_CACHE) or (
                            AUXILLARY_CACHE is not None
                            and len(AUXILLARY_CACHE) > 0 and cache_validate(
                                record['cacheId'], AUXILLARY_CACHE)):
                        continue
                    else:
                        raise IOError(
                            'Cache does not contain spectrogram file',
                            record['cacheId'])
    if MERGE_CACHES:
        merge_cache(PRIMARY_CACHE, AUXILLARY_CACHE)

    D.save(DATASET_TARGET)
    if RECORDS_DUMP:
        shutil.rmtree(RECORDS_DUMP)
