#import numpy as np
#from numpy.core.fromnumeric import mean
#import libs.functions as _fn
#import libs.visualization as _viz
import libs.data as _db
#import os
#import sys
#import datetime
#import librosa
#import soundfile as sf
'''
D = _db.importDataset(DBHOST, DBPORT, "vctk", "train-80-200")
_db.writeDataset(D, './datasets/train-80-200-new.dt')
D = _db.importDataset(DBHOST, DBPORT, "vctk", "validate-80-200")
_db.writeDataset(D, './datasets/validate-80-200-new.dt')
D = _db.importDataset(DBHOST, DBPORT, "vctk", "test-80-200")
_db.writeDataset(D, './datasets/test-80-200-new.dt')
#D = _db.readDataset('./datasets/VCTK-train-80-200.dt')
'''
#print("xxx")
'''means = []
for d in D.values():
    means.append(np.mean(_db.cacheRead(d['cacheId'])))
mean = np.mean(means)
print(mean)'''
'''spgs = np.zeros((len(D), 200, 200))
for i, d in D.items():
    spgs[i] = np.squeeze(_db.cacheRead(d['cacheId']), axis=0)

print(np.mean(spgs))
print(np.std(spgs))'''
'''
D = _db.readDataset('./datasets/voxceleb2-validate-224x224.dt')
#print(_db.recordsAvailable(D))
_db.flag_selected(D, 999)
print(_db.recordsAvailable(D))
'''
'''
print(_db.getUniqueSpeakers(D))
print(len(_db.getUniqueSpeakers(D)))
'''

#D = _db.readDataset('./datasets/VCTK-validate-augmentations.dt')
#for i in range(len(D)):
#	D[i]['cacheId'] = D[i]['speaker'] + '-' + D[i]['cacheId']
#_db.writeDataset(D, 'VCTK-validate-80x200-upd.dt')
'''

fname = "/home/my3bikaht/vctk-corpus-0.92/test/p225/p225_212_mic1.flac"

data, sr = librosa.load(fname)

y = _fn.whiteNoise(data, 7)
sf.write(f'wn7dB.wav', y, sr, 'PCM_24')
y = _fn.whiteNoise(data, 12)
sf.write(f'wn12dB.wav', y, sr, 'PCM_24')

y = _fn.timeStretch(data, 0.8)
sf.write(f'ts08.wav', y, sr, 'PCM_24')
y = _fn.timeStretch(data, 1.2)
sf.write(f'ts12.wav', y, sr, 'PCM_24')

y = _fn.pitchShift(data, -2)
sf.write(f'psn2.wav', y, sr, 'PCM_24')
y = _fn.pitchShift(data, 3)
sf.write(f'psp3.wav', y, sr, 'PCM_24')

y = _fn.addMusic(data, snr=10)
sf.write(f'add.wav', y, sr, 'PCM_24')

for i in range(5):
    y = _fn.addAmbient(data, snr=6)
    sf.write(f'amb{i}.wav', y, sr, 'PCM_24')

'''

#arguments = [arg.replace('-', '') for arg in sys.argv][1:]
#print (f'The script is called with arguments: {arguments}')
#if 'wandb' in arguments:
#	print('YAS!')
'''
#speakers = ['p231', 'p232', 'p340', 'p345']
speakers = ['MAFM0', 'MTPF0', 'MRLJ1']
D = _db.readDataset('./datasets/TIMIT-train-full.dt')
keys = []
for i in range(len(D)):
    for speaker in speakers:
        if D[i]['speaker'] == speaker:
            keys.append(i)

print(f'Removing {len(keys)} records')
for key in keys:
    del D[key]

_db.writeDataset(D, './datasets/TIMIT-train-full-clean.dt')
'''

#D = _db.readDataset('./datasets/TIMIT-train-full.dt')
#speakers_filter = _db.getSpeakersSubset(D)
#Da = _db.getRandomizedSubset(D, speakers_filter=speakers_filter)
#Db = _db.getRandomizedSubsetWithAugmentation(D, speakers_filter=speakers_filter)
#Dc = _db.getRandomizedSubsetWithAugmentation(D, speakers_filter=speakers_filter, augmentations_filter=[0])
#print("asd")
'''
f1 = '/mnt/nvme2tb/datasets/vctk/__cache_test__/0000000001/p256_112.wav_0'
f2 = '/mnt/nvme2tb/datasets/vctk/__cache_full2__/0000000001/p225_082.wav_0'

a = np.load(f1)
b = np.load(f2)

print(type(a))
print(type(b))
'''
#import os
#indices = [5146, 5154, 5156, 5160, 5164, 5173, 5177, 5189, 5191, 5200, 5201, 5203]
#D = _db.readDataset('./datasets/TIMIT-padded-simple/TIMIT-validate.dt')
#for i in indices:
#	print(D[i]['sample'])

#storage = "/home/my3bikaht/datasets/timit/TIMIT2/__cache_padded_simple__/0000000004"
#cacheId = 'MDWH0_SI665.WAV.wav_0'
#with open(os.path.join(storage, cacheId), 'rb') as f:
#	arr = np.load(f, allow_pickle=True)
#a = _db.cacheRead(d['cacheId'], )
'''
del D[5146]
del D[5154]
del D[5156]
del D[5160]
del D[5164]
del D[5173]
del D[5177]
del D[5189]
del D[5191]
del D[5200]
del D[5201]
del D[5203]
_db.writeDataset(D, './datasets/TIMIT-padded-simple/TIMIT-validate-fix2.dt')

D = _db.readDataset('./datasets/TIMIT-padded-simple/TIMIT-validate-fix2.dt')
'''
D = _db.readDataset('./datasets/vox2-3s-train.dt')
sum = 0
for batch_speakers in _db.speakersIterator(D, 32, True):
    sum += len(batch_speakers)
    print(sum, len(batch_speakers), batch_speakers)
'''
for index, d in D.items():
#	if d['cacheId'] == 'MDWH0_SI665.WAV.wav_0':
#		print(index)
    try:
        a = _db.cacheRead(d['cacheId'], "/home/my3bikaht/datasets/TIMIT2/__cache_padded_simple__")
    except:
        print(index)
        continue
#'''
