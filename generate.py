import numpy as np
import libs.functions as _fn
from libs.data import Dataset
import os
from cv2 import resize
import multiprocessing as mp
from datetime import datetime
#import soundfile as sf
#import librosa


def time(message=""):
    pass
    #print(datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'), ":",
    #      message)


'''
MEDIA_DIR = "/media/my3bikaht/EXT4/datasets/TIMIT2/sorted/train"
DATASET_TARGET = "/media/my3bikaht/EXT4/datasets/TIMIT2/datasets/glued-2s-mel80-tokenized-plain-train.dt"
SPECTROGRAMS_CACHE = "/media/my3bikaht/EXT4/datasets/TIMIT2/cache/glued-2s-mel80-tokenized-plain"
'''
MEDIA_DIR = "/media/sergey/3Tb1/datasets/datasets/ambient_noises/raw"
DATASET_TARGET = "/media/sergey/3Tb1/datasets/datasets/ambient_noises/ambient.dt"
SPECTROGRAMS_CACHE = "/media/sergey/3Tb1/datasets/datasets/ambient_noises/cache"

MUSICLIBDIR = "/home/my3bikaht/datasets/music-genres-kaggle/Data/genres_original"
AMBIENTLIBDIR = "/home/my3bikaht/datasets/ambient_noises"

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

# Add white noise to sound during generation
ADDWHITENOISE = False
SNR_OPTIONS = [15]

# Add time stretch to sound during generation
ADDTIMESTRETCH = False
TS_OPTIONS = [1.2, 0.8]

# Add pitch shift to sound during generation
ADDPITCHSHIFT = False
PS_OPTIONS = [-2, 3]

# Add music to sound during generation
ADDMUSIC = False
MUS_OPTIONS = [7]

# Add ambient to sound during generation
ADDAMBIENT = False
AMB_OPTIONS = [7]

# Add number of random shifts for non-zero content
# (only applicable this slice was padded with zeros)
ADDRANDOMSHIFT = False
RANDOMSHIFTS = 3

#SPECTROGRAMS#
MAX_SPEAKERS = 6000
MAX_SAMPLES_PER_SPEAKER = 20
MAX_SPECTROGRAMS_PER_SAMPLE = 5
PICK_RANDOM_SPECTROGRAMS = True


def multiprocess_sample(sample):
    # For each sample (file) let's generate
    # set of spectrograms based on length and steps
    #audioEmpzd = _fn.preemphasize(sample[1])
    audioEmpzd = sample[1]
    spectrograms = _fn.generate_slices(audioEmpzd,
                                       length=SLICE_MS / 1000,
                                       sr=16000,
                                       step=STEP_MS / 1000,
                                       trim=TRIM_MS / 1000,
                                       strategy=SLICING_STRATEGY,
                                       min_len=SKIPSHORTSLICES,
                                       n_mels=MEL_BANKS,
                                       augm=0)

    if spectrograms is None:
        return None
    '''
    origin = slices_raw.copy()
    slices = [(el, 0) for el in slices_raw]
    if ADDWHITENOISE is True:
        for noise_option in SNR_OPTIONS:
            whitenoised = []
            for signal in origin:
                whitenoised.append((_fn.whitenoise(signal, noise_option), 1))
            slices.extend(whitenoised)
    if ADDTIMESTRETCH is True:
        for ts_option in TS_OPTIONS:
            timestretched = []
            for signal in origin:
                timestretched.append((_fn.timestretch(signal, ts_option), 2))
            slices.extend(timestretched)
    if ADDPITCHSHIFT is True:
        for ps_option in PS_OPTIONS:
            pitchshifted = []
            for signal in origin:
                pitchshifted.append((_fn.pitchshift(signal,
                                                    n_steps=ps_option), 3))
            slices.extend(pitchshifted)
    if ADDAMBIENT is True:
        for amb_option in AMB_OPTIONS:
            addambed = []
            for signal in origin:
                addambed.append((_fn.add_ambient(signal, snr=amb_option), 4))
            slices.extend(addambed)
    if ADDMUSIC is True:
        for mus_option in MUS_OPTIONS:
            addmusiced = []
            for signal in origin:
                addmusiced.append((_fn.add_music(signal, snr=mus_option), 5))
            slices.extend(addmusiced)
    if ADDRANDOMSHIFT is True:
        for i in range(RANDOMSHIFTS):
            addrandomshifted = []
            for signal in origin:
                shifted = _fn.add_randomshift(signal)
                if shifted is not None:
                    addrandomshifted.append((shifted, 6))
            slices.extend(addrandomshifted)
    '''
    #for i, slice in enumerate(slices):
    #sf.write(f'./soundcheck/record{i}.ogg', slice, 16000, format='OGG')

    #shuffle slices to be able to pick spectrograms within sample in random order
    indices = np.arange(len(spectrograms))
    if PICK_RANDOM_SPECTROGRAMS:
        np.random.shuffle(indices)
        spectrograms = [spectrograms[i] for i in indices]
    inner_counter = 0
    records = []
    for (spectrogram, augm), eachindex in zip(spectrograms, indices):
        if inner_counter == MAX_SPECTROGRAMS_PER_SAMPLE:
            break
        spectrogram += 40
        spectrogram /= 40
        if FORCE_SHAPE:
            spectrogram_resized = resize(spectrogram,
                                         dsize=FORCE_SHAPE_SIZE)[np.newaxis]
        else:
            spectrogram_resized = spectrogram[np.newaxis].astype(np.float32)

        cache_id = str(sample[0]) + "_" + str(eachindex.item())
        records.append({
            'sample': sample[0],
            'position': eachindex.item(),
            'cacheId': cache_id,
            'spectrogram': spectrogram_resized,
            'augmentation': augm
        })
        inner_counter += 1
    return records if len(records) > 0 else None


_fn.report(" ************************************************** ")
_fn.report(" **            Spectrograms generation           ** ")
_fn.report(" ************************************************** ")

_fn.report("----------------- CURRENT CONFIG -----------------")
_fn.report(f'Slicing strategy: {SLICING_STRATEGY}')
_fn.report(f'Media dir: {MEDIA_DIR}')
_fn.report(f'Dataset target: {DATASET_TARGET}')
_fn.report(f'Cache dir: {SPECTROGRAMS_CACHE}')
_fn.report(f'Slice size, ms {SLICE_MS}')
_fn.report(f'Step size, ms {STEP_MS}')
_fn.report(f'Using up to {MAX_SAMPLES_PER_SPEAKER} samples per speaker')
_fn.report(
    f'   with up to {MAX_SPECTROGRAMS_PER_SAMPLE} spectrograms per sample')
_fn.report(f'   for up to {MAX_SPEAKERS} speakers,')
_fn.report(f'   randomly selected (?): {PICK_RANDOM_SPECTROGRAMS}')
_fn.report(f'Melbanks: {MEL_BANKS}')
_fn.report(f'Trimming audio, ms: {TRIM_MS}')
_fn.report(f'Skipping first frame: {SKIPFIRSTFRAME}')
_fn.report(
    f'Augmentations: WN: {ADDWHITENOISE}, TS: {ADDTIMESTRETCH} PS: {ADDPITCHSHIFT} MSC: {ADDMUSIC} AMB: {ADDAMBIENT} RND: {ADDRANDOMSHIFT}'
)
input("Press any key to continue >> ")

folders = [f.path for f in os.scandir(MEDIA_DIR) if f.is_dir()]
if MAX_SPEAKERS > 0:
    folders = folders[:MAX_SPEAKERS]
'''
# This code prints all keys in collection
# But I can't recall why I added it :)
map = Code("function() { for (var key in this) { emit(key, null); } }")
reduce = Code("function(key, stuff) { return null; }")
result = dbCollectionTrain.map_reduce(map, reduce, "myresults")
print(result.distinct('_id'))
'''
D = Dataset(cache_path=SPECTROGRAMS_CACHE)
if ADDAMBIENT:
    _fn.load_ambient_to_memcache(AMBIENTLIBDIR)
    _fn.report("Loading ambient samples to memory cache completed")
if ADDMUSIC:
    _fn.load_music_to_memcache(MUSICLIBDIR)
    _fn.report("Loading music samples to memory cache completed")

pool = mp.Pool(mp.cpu_count())
for i, folder in enumerate(folders):
    _fn.report("Processing speaker", os.path.basename(folder), ":", (i + 1),
               "out of", len(folders))
    time("loop start")
    speaker = os.path.basename(folder)
    samples = _fn.raw_audio_by_dir(folder, MAX_SAMPLES_PER_SPEAKER)
    time("raw audio loaded")
    if len(samples) > 0:
        #for sample in samples:
        #	mp_results = multiprocess_sample(sample)
        mp_results = pool.map(multiprocess_sample, samples)
    else:
        _fn.report("No audio examples fit requirements for speaker", speaker)
    if all([el is None for el in mp_results]):
        _fn.report("Unable to generate spectrograms (too short) for speaker",
                   speaker)
    else:
        time("spectrograms generated")
        for results in mp_results:
            if results is not None:
                for result in results:
                    D.cache_write(result['cacheId'], result['spectrogram'])
                    D.append({
                        'speaker': speaker,
                        'sample': result['sample'],
                        'position': result['position'],
                        'cacheId': result['cacheId'],
                        'augmentation': result['augmentation'],
                        'selected': False,
                        'embedding': None
                    })
    time("spectrograms saved")
    with open('processed.log', 'a') as f:
        f.write(f'{speaker}, {i}\n')
D.save(DATASET_TARGET)
