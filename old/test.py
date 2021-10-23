import random
import torch
import numpy as np

import libs.functions as _fn
import libs.data as _db
import libs.validation as _val

#from libs.models.tripletCustomResNet34 import TripletModel
from libs.models.tripletCNNmodelVCTK import TripletModel

CHECKPOINTFILE = "./checkpoints/CNN02.dict"

# Fixing seed for reproducibility
'''
fixedSeed = 1
torch.manual_seed(fixedSeed)
torch.cuda.manual_seed(fixedSeed)
np.random.seed(fixedSeed)
random.seed(fixedSeed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
'''

_fn.report("**************************************************")
_fn.report("**                Testing script                **")
_fn.report("**************************************************")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
_fn.report("Torch is using device:", device)

checkpoint = torch.load(CHECKPOINTFILE)

model = TripletModel()
model.float()
if torch.cuda.is_available():
    model.cuda()

model.load_state_dict(checkpoint['state_dict'])
_fn.report("Model state dict loaded from latest checkpoint")

T = _db.readDataset(DATASET_DIR + DATASET_TEST)

_fn.report("Updating embeddings")
model.eval()
_db.updateEmbeddings(T, model=model, device=device)
model.train()

centroids = checkpoint['centroids']

top1, top5, total = 0., 0., 0
_db.reset(T)
speakers = _db.getUniqueSpeakers(T)
for speaker in speakers:
    samples = _db.getSamples(T, speaker)
    for sample in samples:
        records = _db.getRecords(T, sample)
        t1, t5 = _val.sequential_probability(candidates=records,
                                             centroids=centroids,
                                             truth=speaker)
        top1 += int(t1)
        top5 += int(t5)
        total += 1
print(top1, top5, total, float(top1 / total), float(top5 / total))
