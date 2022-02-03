import os
import librosa
import librosa.display
import libs.functions as _fn
#import torch
import numpy as np
#from libs.models.janet_vox_v2a import Janet
#from collections import Counter
#import warnings
#import tqdm
import random
import soundfile as sf
#import noisereduce as nr

from matplotlib import pyplot as plt

filenames = os.listdir('./mispredicted')
filename = random.choice(filenames)
#random.shuffle(filenames)


def echo(signal, offset=100, drop=0.8, sr=16000):
    eo = (signal * (1 - drop))[int(offset * sr / 1000):]
    eo = np.concatenate((np.zeros(len(signal) - len(eo)), eo))
    signal = signal * drop
    output = signal + eo
    return output


def echoes(signal, repeats=10):
    for i in range(repeats):
        signal = echo(signal)
    return signal


def save_spg(img, filename):
    librosa.display.specshow(img)
    plt.savefig(filename)


def spg_and_save(signal, filename):
    spg = _fn.generate_spectrogram(signal, n_mels=64)
    spg += 40
    spg /= 40
    save_spg(spg, filename)
    return spg


plt.axis('off')
audio, sr = librosa.load(os.path.join('./mispredicted', filename),
                         sr=16000,
                         mono=True)
sf.write('original.wav', audio, samplerate=16000)
#spg_original = spg_and_save(audio, 'original.png')

#
#board = Pedalboard([
#    Compressor(threshold_db=-25, ratio=25),
#], sample_rate=sr)
#sf.write('compressor25-25.wav', board(audio), samplerate=16000)

sf.write('fitered.wav', board(audio), samplerate=16000)
print(filename)
'''
        Compressor(threshold_db=-50, ratio=25),
    Gain(gain_db=30),
    Chorus(),
    LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=900),
    Phaser(),
    '''

#spg_modified = spg_and_save(board(audio), 'echo1.png')
'''
spg_diy = spg_echo(spg_original, 25, 5, .9)
#spg_diy += 40
#spg_diy /= 40
save_spg(spg_diy, 'diy.png')
diff = spg_diy - spg_modified
save_spg(diff, 'diff.png')
print(f'max:{np.max(diff)}, mean:{np.mean(diff)}')
'''