import torch
import torchvision
import numpy as np
import pymongo

import libs.functions as _fn
import libs.data as _db

_fn.report("Connecting to MongoDB")
from pymongo.errors import ConnectionFailure

client = pymongo.MongoClient('localhost', 27017)
try:
    client.admin.command('ismaster')
except ConnectionFailure:
    print("Mongo server not available")
db = client['mnist']
trainDataset = db['train-2k-28']
validateDataset = db['validate-2k-28']

batch_size_train = 1
batch_size_test = 1

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        '../../datasets',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
    batch_size=batch_size_train,
    shuffle=False)

limit = 2000
for counter, (data, label) in enumerate(train_loader):
    if counter >= limit:
        break
    newId = str(label.item()) + '_' + str(counter)
    _db.cacheWrite(newId,
                   torch.squeeze(data, 0).numpy(),
                   cache='./__mnist-cache__/')
    record = {
        "speaker": str(label.item()),
        "sample": '0',
        "cacheId": newId,
        "position": 0,
        "selected": False,
        "embedding": None
    }
    target = np.random.random()
    if target > .2:
        result = trainDataset.insert_one(record)
    else:
        result = validateDataset.insert_one(record)
