import os
import datetime
#import soundfile as sf
import numpy as np
import librosa
import librosa.display
from numpy.lib import split
import torch
from scipy.spatial.distance import cosine
import warnings
import random
import math
from string import ascii_lowercase
from scipy.spatial.distance import cdist
import libs.visualization as _viz
import matplotlib.pyplot as plt
import sys
from glob import glob

global ambientmemcache
global musicmemcache
musicmemcache = {}
ambientmemcache = {}


# output current timestamp with message to console
def timestamp():
    st = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S |')
    return st


# output reported arguments to log file
def report(*args):
    string = timestamp()
    for arg in args:
        if isinstance(arg, str):
            string += ' ' + arg
        else:
            string += ' ' + str(arg)
    print(string)
    log = open("run.log", "a+")
    log.write(string + "\n")
    log.close()
    return True


def get_random_hash():
    return ''.join(random.choice(ascii_lowercase) for i in range(10))


def fix_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.deterministic = True
    return True


def cleanup(run_id=None, checkpointdir='./checkpoints'):
    if run_id is None:
        raise Exception("No run id provided")
    allcp = glob(os.path.join(checkpointdir, run_id + "*"))
    latestcp = max(allcp, key=os.path.getctime)
    for cp in allcp:
        if cp == latestcp:
            continue
        else:
            os.remove(cp)
    print(
        f"Cleanup complete, removed all checkpoints except {os.path.basename(latestcp)}"
    )
    return True


def early_stop(losses=None, epochs=20, criterion='max'):
    if losses is None:
        return False
    if type(losses) == torch.Tensor:
        losses = losses.detach().cpu().numpy()
    if not type(losses) == np.ndarray:
        losses = np.array(losses)
    crit_fn = np.argmax if criterion == 'max' else np.argmin
    best = losses.shape[0] - crit_fn(np.flip(losses), axis=0) - 1
    if best <= losses.shape[0] - epochs:
        return True
    else:
        return False


def todolist():
    if os.path.isfile('./TODO'):
        report('********************* TODO REMINDER *********************')
        with open('./TODO') as f:
            lines = f.readlines()
            for line in lines:
                report(f'***** {line.rstrip()}')
        report('*********************************************************')


def raw_audio_by_dir(folder=".", limit=0):
    output = []
    for root, _, filenames in os.walk(folder):
        for index, filename in enumerate(filenames):
            if limit > 0 and index >= limit:
                continue  # skip rest of the files
            checkfilename = root + '/' + filename
            #print(os.path.basename(root))
            extensions = ('.mp3', '.wav', '.mp4', ".ogg", '.flac', '.aac',
                          '.m4a')
            if checkfilename.endswith(extensions):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data, sr = librosa.load(checkfilename)
                except Exception:
                    report("Can't read file %s, skipping" % (checkfilename))
                    continue
                    #raise IOError("Can't read file", checkfilename)
                    #return np.empty(shape=(0, 0))
                if data.shape[0] == 0:
                    report(
                        "Warning: Audio row returned 0 length when trying to read",
                        checkfilename, "(skipping)")
                    continue
                else:
                    # If multichannel then convert to mono by averaging between channels
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        data = librosa.resample(data, sr, 16000)
                        #output.append(
                        #    [os.path.basename(root) + "-" + filename, data])
                        output.append([filename, data])
    return output


def preemphasize(raw_audio, pre_emphasis=0.97):
    output = np.append(raw_audio[0],
                       raw_audio[1:] - pre_emphasis * raw_audio[:-1])
    return output


def nonpaddedsignal(signal):
    zeros = np.nonzero(np.flip(signal))[0][0]
    return signal[:-zeros] if zeros > 0 else signal


# add white noise
def whitenoise(signal, snr=10):
    ss = nonpaddedsignal(signal)
    rms = np.mean(librosa.feature.rms(y=ss))
    if rms == 0:
        raise Exception(
            "RMS mean is equal to zero. Either you passed zeros as signal or librosa malfunctioned"
        )
    # generate rms of noise then add
    noise = np.random.normal(0, np.sqrt(rms**2 / 10**(snr / 10)), ss.shape[0])
    return np.pad(ss + noise, (0, len(signal) - len(ss)))


# stretch signal over time without changing pitch
def timestretch(signal, rate=1):
    output = librosa.effects.time_stretch(signal, rate)
    if output.shape[0] > signal.shape[0]:
        output = output[:signal.shape[0]]
    elif output.shape[0] < signal.shape[0]:
        output = np.pad(output, (0, signal.shape[0] - output.shape[0]))
    else:
        output = signal
    return output


# add pitch shift in the background
def pitchshift(signal, n_steps=0):
    output = librosa.effects.pitch_shift(y=signal, sr=16000, n_steps=n_steps)
    return output


def load_ambient_to_memcache(ambient_dir):
    wavs = []
    for root, dirnames, filenames in os.walk(ambient_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                wavs.append(root + '/' + filename)
    for wav in wavs:
        sample, _ = librosa.load(wav, sr=16000, mono=True)
        pos = len(ambientmemcache) + 1
        ambientmemcache[pos] = sample
    return True


class Status():
    def __init__(self) -> None:
        self.first_call = True

    def print(self, s):
        if self.first_call:
            print(s)
            self.first_call = False
        else:
            CURSOR_UP_ONE = '\033[K'
            ERASE_LINE = '\x1b[2K'
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE + '\r')
            print(s)


def load_music_to_memcache(music_dir):
    wavs = []
    for root, dirnames, filenames in os.walk(music_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                wavs.append(root + '/' + filename)
    for wav in wavs:
        sample, _ = librosa.load(wav, sr=16000, mono=True)
        pos = len(musicmemcache) + 1
        musicmemcache[pos] = sample
    return True


def reverse_sample(signal):
    return signal[::-1]


# add talk in the background
def add_ambient(signal, snr=15):
    ss = nonpaddedsignal(signal)
    #if len(ambientmemcache) == 0:
    #	loadAmbientToMemCache() NOT THE BEST OPTION FOR MULTIPROCESSING
    key = random.choice(list(ambientmemcache.keys()))
    noise = ambientmemcache[key]
    ln = len(noise)
    ls = len(ss)
    if ln < ls:
        mult = math.ceil(ls / ln)
        noise = np.tile(noise, mult)
    shiftmax = len(noise) - len(ss)
    startpos = np.random.randint(shiftmax)
    noise = noise[startpos:startpos + len(ss)]
    rms_ss = np.mean(librosa.feature.rms(y=ss))
    rms_noise = np.mean(librosa.feature.rms(y=noise))
    target_rms_noise = rms_ss / (10**(snr / 10))
    scale = target_rms_noise / rms_noise
    return np.pad(ss + noise * scale, (0, len(signal) - len(ss)))


# add music in the background
def add_music(signal, snr=10):
    ss = nonpaddedsignal(signal)
    #if len(ambientmemcache) == 0:
    #	loadAmbientToMemCache() NOT THE BEST OPTION FOR MULTIPROCESSING
    key = random.choice(list(musicmemcache.keys()))
    noise = musicmemcache[key]
    ln = len(noise)
    ls = len(ss)
    if ln < ls:
        mult = math.ceil(ls / ln)
        noise = np.tile(noise, mult)
    shiftmax = len(noise) - len(ss)
    startpos = np.random.randint(shiftmax)
    noise = noise[startpos:startpos + len(ss)]
    rms_ss = np.mean(librosa.feature.rms(y=ss))
    rms_noise = np.mean(librosa.feature.rms(y=noise))
    target_rms_noise = rms_ss / (10**(snr / 10))
    scale = target_rms_noise / rms_noise
    return np.pad(ss + noise * scale, (0, len(signal) - len(ss)))


def add_randomshift(signal):
    #check if non-zero elements available end of array
    zeros = np.nonzero(np.flip(signal))[0][0]
    if zeros > 0:
        shift = np.random.randint(zeros)
        return np.roll(signal, shift)
    else:
        return None


def prepare_audio(raw_audio,
                  sr=16000,
                  length=2.,
                  step=.1,
                  trim=.0,
                  min_len=0,
                  sample_name=None):
    minlenraw = int(min_len * sr * length)
    lenraw = int(sr * length)
    # stepraw = int(sr * step)
    trimraw = int(sr * trim)

    # trimming start and end of the record
    # because in case of voxceleb there are multiple voices
    # prob due to algo
    if len(raw_audio) <= trimraw * 2:
        return None
    if trim > 0:
        raw_audio = raw_audio[trimraw:-trimraw]

    # trim silence from raw audio first
    raw_audio = librosa.effects.trim(y=raw_audio, top_db=30)[0]

    # new strategy:
    # - slice raw audio into intervals
    # - replace silent pauses with zeros (for attention mask) (because pauses matter)
    # - generate one large spectrogram (speed up process)
    # - retrieve parts of spectrogram less than 3sec long starting from each "word" (slices attached to actual words)
    # - (or we can make sets of "words" with set length, like two "words" in a row)
    # - pad rest of the spectrogram with "zeros"

    #split into intervals
    intervals = librosa.effects.split(y=raw_audio, top_db=30)
    # pad skipped parts with zeros
    raw_audio_zeroed = np.zeros((raw_audio.shape), dtype='float32')
    for interval in intervals:
        raw_audio_zeroed[interval[0]:interval[1]] = raw_audio[
            interval[0]:interval[1]]

    if len(raw_audio_zeroed) < minlenraw:
        return None
    elif len(raw_audio_zeroed) < lenraw:
        raw_audio_zeroed = np.pad(raw_audio_zeroed,
                                  (0, lenraw - len(raw_audio_zeroed)),
                                  'constant')

    return raw_audio_zeroed


def pad(data, target_len):
    diff = target_len - data.shape[0]
    if diff > 0:
        pad = np.zeros((diff, data.shape[1]))
        data = np.concatenate((data, pad))
    return data


def trim(data, target_len):
    diff = target_len - data.shape[0]
    if diff < 0:
        data = data[:target_len, :]
    return data


def generate_slices(raw_audio,
                    sr=16000,
                    length=2.,
                    step=.1,
                    trim=.0,
                    strategy='glue',
                    min_len=0,
                    n_mels=40,
                    frame_length_ms=25,
                    hop_ms=10,
                    augm=0):
    min_len_spg = int(min_len * 1000 / hop_ms)
    step_spg = int(step * 1000 / hop_ms)
    target_shape = (int(length * 1000 / hop_ms), n_mels)

    # trimming start and end of the record
    # because in case of voxceleb there are multiple voices
    # prob due to algo
    if len(raw_audio) <= int(sr * trim * 2):
        return None
    if trim > 0:
        raw_audio = raw_audio[int(sr * trim):-int(sr * trim)]

    # trim silence from raw audio first
    intervals = librosa.effects.split(y=raw_audio, top_db=30)
    # we have multiple options here:
    # A. Glue intervals together into one big interval
    # B. Pad intervals with zeros
    # Option B has multiple issues:
    # - do we pad on the lelf and right both or just right?
    # - should we shift non-zero sequences across slice as an augmentation strategy?
    data = []
    if strategy == 'pad':
        # Split audio, then pad right each, then slice large ones
        for interval in intervals:
            if interval[1] - interval[0] < 400:  # n_fft = 400
                continue
            spg = generate_spectrogram(raw_audio[interval[0]:interval[1]], sr,
                                       n_mels, frame_length_ms, hop_ms).T
            if spg.shape[0] < min_len_spg:
                continue
            else:
                data.append(pad(spg, target_shape[0]))

    elif strategy == 'glue' or strategy == 'tokenize':
        # split, then combine into one, then slice
        data = np.empty(shape=(0, n_mels), dtype=np.float32)
        for i, interval in enumerate(intervals):
            if interval[1] - interval[0] < 400:  # n_fft = 400
                continue
            spg = generate_spectrogram(raw_audio[interval[0]:interval[1]], sr,
                                       n_mels, frame_length_ms, hop_ms).T
            data = np.concatenate((data, spg))
            # if 'tokenize' insert token
            if strategy == 'tokenize':
                if i + 1 < len(intervals):
                    data = np.concatenate((data, (np.zeros((1, n_mels)) - 80)))
                    # add column of -80, lowest value

        # If combined sound file length less than 50% of window size
        # then discard else pad with zeros on the right
        if data.shape[0] < min_len_spg:
            return None
        else:
            data = [pad(data, target_shape[0])]

    output = []
    for spg in data:
        num_slices = int((spg.shape[0] - target_shape[0]) / step_spg) + 1
        slices = np.empty(shape=(num_slices, target_shape[0], n_mels),
                          dtype=np.float32)
        for i in range(num_slices):
            slices[i] = spg[int(i * step_spg):int(i * step_spg +
                                                  target_shape[0])]
        for slice in slices:
            output.append((slice.T, augm))

    return output


def generate_spectrogram(raw_audio,
                         sr=16000,
                         n_mels=40,
                         frame_length_ms=25,
                         hop_ms=10):
    # 512 (32ms) vs 256 (16ms) when sr=16000 / optimal for speech is (512) 23ms with sr=22050
    # "frame length=25ms, frame shift=10ms" / 1/16000 => window = 400, hop = 160
    frame_length = int(frame_length_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    S = np.abs(
        librosa.stft(y=raw_audio,
                     n_fft=frame_length,
                     hop_length=hop,
                     window='hamming',
                     center=True))**2
    # Calculating mel_basis
    mel_basis = librosa.filters.mel(sr=sr,
                                    n_fft=frame_length,
                                    fmin=20,
                                    fmax=8000,
                                    htk=True,
                                    n_mels=n_mels)
    F = np.dot(mel_basis, S)
    Fdb = librosa.power_to_db(F, ref=np.max)[:, :-1]

    # plt.clf()
    # plt.plot(raw_audio)
    # plt.savefig('E.png')
    # plt.clf()
    # librosa.display.specshow(output,
    #                          sr=sr,
    #                          hop_length=hop,
    #                          x_axis='time',
    #                          y_axis='mel')
    # plt.savefig('F.png')
    return Fdb


def distance(vector1, vector2):
    if torch.is_tensor(vector1) is True:
        vector1 = vector1.detach().cpu().numpy()
        vector2 = vector2.detach().cpu().numpy()
    return cosine(vector1, vector2)


def centroid_distances(centroids, filename=None):
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


def generate_distance_matrix_and_dump(centroids,
                                      filename="centroid_distances.csv"):
    dm = cdist(np.asarray([c[1] for c in centroids]),
               np.asarray([c[1] for c in centroids]),
               metric='cosine')
    zeroed = dm * (1 - np.tri(dm.shape[0], dm.shape[1], k=0))
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
