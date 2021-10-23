import numpy as np
from numpy.core.fromnumeric import mean
import libs.functions as _fn
import libs.visualization as _viz
import libs.data as _db
from libs.models.tripletCustomResNet34 import TripletModel
import torch

CHECKPOINTFILE = "./checkpoints/dict_henujbceje_ResNet50_epoch199.dict"
checkpoint = torch.load(CHECKPOINTFILE)
model = TripletModel()
model.float()
model.cuda()
model.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda:0")

D = _db.readDataset('./datasets/VCTK-train-224x224.dt')
speakers = _db.getUniqueSpeakers(D)

model.eval()
_db.updateEmbeddings(D, model, device)

for speaker in speakers:
    records = _db.getSiblings(D, speaker, limit=-1, flag=False)
    embeddings = [el['embedding'] for el in records]
    meanOf = np.mean(embeddings, axis=0)
    distances = [_fn.distance(el, meanOf) for el in embeddings]
    for distance in distances:
        distance**2
    radiusSTD = np.sqrt(
        np.sum([el**2 for el in distances]) / (len(distances) - 1))
    relativeDistances = distances - (radiusSTD * 3)
    outliers = [records[i] for i in (relativeDistances > 0).nonzero()[0]]
    output = [{
        '_id': el['_id'],
        'speaker': el['speaker'],
        'sample': el['sample'],
        'position': el['position'],
        'cacheId': el['cacheId']
    } for el in outliers]
    print("-----------------------------------------------")
    print("Outliers:", output)
    for el in outliers:
        _viz.graph(_db.cacheRead(el['cacheId']), 'deleteme.png')
        input(">>>")
