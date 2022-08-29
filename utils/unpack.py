from libs.data import Dataset
import os
import pickle

# Use this to unpack batch saved spectrograms if used during generating

SOURCE_DIR = '/media/sergey/3Tb1/cache'
TARGET_DIR = '/mnt/nvme2tb/datasets/voxceleb2/double/cache'
D = Dataset(filename='/mnt/nvme2tb/datasets/voxceleb2/double/datasets/test.dt')

sources = os.listdir(SOURCE_DIR)

for source in sources:
    with open(os.path.join(SOURCE_DIR, source), 'rb') as f:
        data = pickle.load(f)
        for fname, record in data.items():
            D.cache_write(fname, record, TARGET_DIR)
