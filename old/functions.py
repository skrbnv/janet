import os
import datetime
#import soundfile as sf
import numpy as np
import librosa
import torch
from scipy.spatial.distance import cosine
import warnings
import random
import math
from string import ascii_lowercase
from scipy.spatial.distance import cdist
import libs.visualization as _viz

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
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    return True


def todolist():
    if os.path.isfile('./TODO'):
        report('********************* TODO REMINDER *********************')
        with open('./TODO') as f:
            lines = f.readlines()
            for line in lines:
                report(f'***** {line.rstrip()}')
        report('*********************************************************')


def rawAudioByDir(folder=".", limit=0):
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
def whiteNoise(signal, snr=10):
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
def timeStretch(signal, rate=1):
    output = librosa.effects.time_stretch(signal, rate)
    if output.shape[0] > signal.shape[0]:
        output = output[:signal.shape[0]]
    elif output.shape[0] < signal.shape[0]:
        output = np.pad(output, (0, signal.shape[0] - output.shape[0]))
    else:
        output = signal
    return output


# add pitch shift in the background
def pitchShift(signal, n_steps=0):
    output = librosa.effects.pitch_shift(y=signal, sr=16000, n_steps=n_steps)
    return output


def loadAmbientToMemCache(AMBIENTLIBDIR):
    wavs = []
    for root, dirnames, filenames in os.walk(AMBIENTLIBDIR):
        for filename in filenames:
            if filename.endswith('.wav'):
                wavs.append(root + '/' + filename)
    for wav in wavs:
        sample, _ = librosa.load(wav, sr=16000, mono=True)
        pos = len(ambientmemcache) + 1
        ambientmemcache[pos] = sample
    return True


def reverseSample(signal):
    return signal[::-1]


# add talk in the background
def addAmbient(signal, snr=15):
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


def loadMusicToMemCache(MUSICLIBDIR):
    wavs = []
    for root, dirnames, filenames in os.walk(MUSICLIBDIR):
        for filename in filenames:
            if filename.endswith('.wav'):
                wavs.append(root + '/' + filename)
    for wav in wavs:
        sample, _ = librosa.load(wav, sr=16000, mono=True)
        pos = len(musicmemcache) + 1
        musicmemcache[pos] = sample
    return True


# add music in the background
def addMusic(signal, snr=10):
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


def addRandomShift(signal):
    #check if non-zero elements available end of array
    zeros = np.nonzero(np.flip(signal))[0][0]
    if zeros > 0:
        shift = np.random.randint(zeros)
        return np.roll(signal, shift)
    else:
        return None


def prepareAudio(raw_audio,
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


def generateSlices(raw_audio,
                   sr=16000,
                   length=2.,
                   step=.1,
                   trim=.0,
                   min_len=0,
                   sample_name=None):
    minlenraw = int(min_len * sr * length)
    lenraw = int(sr * length)
    stepraw = int(sr * step)
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

    #slices = np.zeros((len(intervals), lenraw))
    #for i in range(len(intervals)):
    #	slices[i] = raw_audio_zeroed[intervals[i,0]:intervals[i,0]+lenraw]
    slices = []
    num_slices = int((len(raw_audio_zeroed) - lenraw) / stepraw)
    for i in range(num_slices):
        slices.append(raw_audio_zeroed[i * stepraw:i * stepraw + lenraw])
    return slices


def generate_slices(raw_audio,
                    sr=16000,
                    length=2.,
                    step=.1,
                    trim=.0,
                    strategy='glue',
                    min_len=0):
    min_length_raw = int(min_len * sr * length)
    length_raw = int(sr * length)
    step_raw = int(sr * step)

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

    if strategy == 'glue':
        # Processing option A:
        data = np.empty(shape=(0, 0), dtype=np.float32)
        for interval in intervals:
            data = np.append(data, raw_audio[interval[0]:interval[1]])
        '''
        if not raw_audio.shape[0] == data.shape[0]:
            print(raw_audio.shape)
            print(data.shape)
        '''
        # If combined sound file length less than 80% of window size
        # then skip or pad with zeros on the right
        if data.shape[0] < length_raw:
            if data.shape[0] < length_raw * .8:
                return None
            else:
                data = np.pad(data, (0, length_raw - data.shape[0]),
                              'constant')

        total_slices = int((data.shape[0] - length_raw) / step_raw) + 1
        output = np.empty(shape=(total_slices, length_raw), dtype=np.float32)
        for index in range(total_slices):
            output[index] = data[int(index * step_raw):int(index * step_raw +
                                                           length_raw)]
    else:  # if strategy == 'pad'
        # Processing option B (pad right):
        chunks = []
        for interval in intervals:
            chunk = np.array(raw_audio[interval[0]:interval[1]])
            if chunk.shape[0] < min_length_raw:
                continue
            if chunk.shape[0] < length_raw:
                chunk = np.pad(chunk, (0, length_raw - chunk.shape[0]),
                               'constant')
            chunks.append(chunk)
        output = []
        for chunk in chunks:
            total_slices = int((chunk.shape[0] - length_raw) / step_raw) + 1
            for index in range(total_slices):
                output.append(chunk[int(index *
                                        step_raw):int(index * step_raw +
                                                      length_raw)])
        output = np.array(output)
    return output


def generate_spectrogram(raw_audio,
                         sr=16000,
                         n_mels=40,
                         frame_length_ms=25,
                         hop_ms=10):
    # 512 (32ms) vs 256 (16ms) when sr=16000 / optimal for speech is (512) 23ms with sr=22050
    # "frame length=25ms, frame shift=10ms" / 1/16000 => window = 400, hop = 160
    #print(raw_audio.shape)
    frame_length = int(frame_length_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    #spectrogram_length = int(raw_audio.shape[0] / hop)
    #print(spectrogram_length)
    # result = np.empty([0,80,int(spectrogram_length)])
    # S = librosa.core.stft(y=raw_audio, n_fft=frame_length, hop_length=hop, window='hamming', center=True)
    S = np.abs(
        librosa.stft(y=raw_audio,
                     n_fft=frame_length,
                     hop_length=hop,
                     window='hamming',
                     center=True))**2
    # (201,141)
    #Sdb = librosa.power_to_db(S, ref = np.max)
    # Calculating mel_basis
    mel_basis = librosa.filters.mel(sr=sr,
                                    n_fft=frame_length,
                                    fmin=20,
                                    fmax=8000,
                                    htk=True,
                                    n_mels=n_mels)
    #L = np.log(np.dot(mel_basis, S))
    #Ldb = librosa.power_to_db(L, ref = np.max)
    F = np.dot(mel_basis, S)
    Fdb = librosa.power_to_db(F, ref=np.max)
    Fdb = Fdb[:, :-1]
    # (40,140)
    #librosa.display.specshow(Fdb, sr=sr, hop_length=hop, x_axis='time', y_axis='mel')
    #plt.savefig('F.png')
    #librosa.display.specshow(Ldb, sr=sr, hop_length=hop, x_axis='time', y_axis='mel')
    #plt.savefig('L.png')
    #S = librosa.feature.melspectrogram(y=raw_audio, sr=sr, n_fft=frame_length, hop_length=hop, power=2.0)
    # We can now transform the spectrogram output to a logarithmic scale
    # by transforming the amplitude to decibels. While doing so we will
    # also normalize the spectrogram so that its maximum represent the
    # 0 dB point.
    # Ca

    # Now we need to normalize/standartize V on every frequency bin (row)
    # Applying zero mean unit variance to bins (row)
    #V = Fdb
    #Vmean = V.mean(axis=1)
    #Vstd = V.std(axis=1)
    #Vtop = V - Vmean[:, np.newaxis]
    #Vbottom = Vstd[:, np.newaxis]
    #if we have standard deviation zero we remove this slice (spectrogram) from set
    #if not np.any(Vbottom==0):
    #	V = Vtop/Vbottom
    #librosa.display.specshow(V, sr=sr, hop_length=hop, x_axis='time', y_axis='mel')
    #plt.savefig('V.png')
    #input(">>>")
    #result = np.vstack((result, V[np.newaxis,...]))
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
