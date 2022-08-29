import torch
import torch.nn as nn
from libs.models3 import WideResNet2810
import libs.functions as _fn
from libs.data import Dataset
import pickle

with open('./stuff/speakers_vox2.dt', 'rb') as f:
    speakers = pickle.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
_fn.report("Torch is using device:", device)

indices = torch.load('./stuff/speaker_indices_timit')

model = WideResNet2810(num_classes=5994).float().to(device)
checkpoint = torch.load('./checkpoints/toezr9nh002.dict')
model.load_state_dict(checkpoint['state_dict'])

T = Dataset(filename='/mnt/nvme2tb/datasets/voxceleb2/final/test.dt',
            cache_paths=[
                '/mnt/nvme2tb/datasets/voxceleb2/final/cache/test',
                '/mnt/nvme1tb/datasets/voxceleb2/final/cache/test'
            ],
            useindex=False)

model.eval()
with torch.no_grad():
    for i in range(1000):
        sample = T.__getitem__(i, 2)
        data = torch.tensor(sample[0]).float().to(device)
        pred = model(data.unsqueeze(0))
        predicted = torch.argmax(pred).item()
        actual = speakers[sample[2]['speaker']]
        if predicted == actual:
            print(".", end='')
        else:
            print("x", end='')
            #print(f'({predicted}:{actual})', end='')
