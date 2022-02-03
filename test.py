import os
import librosa
import librosa.display
import libs.functions as _fn
import torch
import numpy as np
from libs.models.janet_vox_v2a import Janet
from collections import Counter
import warnings
import tqdm
#import soundfile as sf
#import noisereduce as nr

warnings.filterwarnings("ignore")

spgdir = '/mnt/nvme2tb/datasets/voxceleb2/sorted/validate'
checkpoint = './checkpoints/16byjx1b012.dict'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
_fn.report("Torch is using device:", device)

model = Janet()
state_dict = torch.load(checkpoint)
model.load_state_dict(state_dict['state_dict'])
model.to(device)

fl = os.listdir(spgdir)
speakers_dirs = []
for el in fl:
    if os.path.isdir(os.path.join(spgdir, el)):
        speakers_dirs.append(el)
speakers = sorted(speakers_dirs)

model.eval()
for speaker in tqdm.tqdm(speakers_dirs):
    fl = os.listdir(os.path.join(spgdir, speaker))
    files = [f for f in fl if f.endswith('.m4a')]
    for f in files:
        audio, sr = librosa.load(os.path.join(spgdir, speaker, f), 16000, True)
        #sf.write('original.wav', audio, samplerate=16000)
        intervals = librosa.effects.split(audio, top_db=30)
        chunks = [audio[el[0]:el[1]] for el in intervals]
        combined = np.concatenate(chunks)

        spg = _fn.generate_spectrogram(combined, n_mels=64)
        spg += 40
        spg /= 40
        pred = []
        for i in range(spg.shape[1] - 192):
            with torch.no_grad():
                y_pred = model(
                    torch.from_numpy(
                        spg[:,
                            i:i + 192]).unsqueeze(0).unsqueeze(0).to(device))
                pred.append(torch.argmax(y_pred).item())
        #print(f'Predictions for speaker {speaker}, sample: {f}')
        counter = Counter(pred)
        best_option = max(counter, key=counter.get)
        if not speakers[best_option] == speaker:
            with open('mispredicted.txt', 'a') as fname:
                fname.write(
                    f'Mispredicted {speakers[best_option]} for {os.path.join(spgdir, speaker, f)}\n'
                )
            #print(f'Misprediction for speaker {speaker}, file {f}')
            #for key, value in counter.items():
            #    print(f'{key} ({speakers[key]}): {value}')
