import librosa
import os
import random
#import soundfile as sf
import tqdm
import string
import librosa.display
import numpy as np
import warnings

warnings.filterwarnings("ignore")

signal_dir_path = '/mnt/nvme2tb/datasets/voxceleb2/sorted/train'
ambient_dir_path = '/mnt/nvme2tb/datasets/music-genres-kaggle/Data/genres_original'

signal_dirs = [f.path for f in os.scandir(signal_dir_path) if f.is_dir()]
ambient_dirs = [f.path for f in os.scandir(signal_dir_path) if f.is_dir()]
#print(signal_dirs)

signal_index = []
for sigdir in signal_dirs:
    signal_index.extend(
        [os.path.join(sigdir, el) for el in os.listdir(sigdir)])
ambient_index = []
for ambdir in ambient_dirs:
    ambient_index.extend(
        [os.path.join(ambdir, el) for el in os.listdir(ambdir)])

#signal_index = os.listdir(signal_dir)
#ambient_index = os.listdir(ambient_dir)


class CachedLoad():
    def __init__(self) -> None:
        self.cache = {}

    def load(self, fname, dir):
        if fname in self.cache.keys():
            return self.cache[fname]
        fpath = os.path.join(dir, fname) if dir is not None else fname
        signal, sr = librosa.load(fpath, mono=True)
        librosa.resample(signal, sr, 16000, fix=True)
        if fname not in self.cache.keys():
            self.cache[fname] = signal
        return signal


def A(raw_audio, sr=16000, n_mels=64, frame_length_ms=25, hop_ms=10):
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
    return F


def Z(F):
    Fdb = librosa.power_to_db(F, ref=np.max)[:, :-1]
    return Fdb


def std(spg):
    return (spg + 40.) / 40.


cache = CachedLoad()
for j in tqdm.tqdm(range(5000)):
    #print(j)

    #print(signal.shape, sr)
    signal = cache.load(random.choice(signal_index), None)
    ambient = cache.load(random.choice(ambient_index), None)
    if ambient.shape[0] > signal.shape[0]:
        start = random.randint(0, ambient.shape[0] - signal.shape[0])
        ambient_cut = ambient[start:start + signal.shape[0]]
    else:
        continue

    mixed = np.add(signal * .9, ambient_cut * .1)

    #sf.write('signal.wav', signal, samplerate=22050)
    #sf.write('ambient.wav', ambient_cut, samplerate=22050)
    #sf.write('mixed.wav', mixed, samplerate=22050)

    spgd = std(Z(A(signal)))
    spgs = std(Z(A(mixed)))
    spgdiff = spgs - spgd
    letters = string.ascii_lowercase
    fname = './diffs-music/' + ''.join(
        random.choice(letters) for i in range(16)) + '.csv'
    np.savetxt(fname, spgdiff, delimiter=",")
