import os
import numpy as np
import tqdm

csv_dir = './diffs-music-louder'
csv_list = os.listdir(csv_dir)
csvs = np.zeros((64, 0))
for f in tqdm.tqdm(csv_list):
    data = np.genfromtxt(
        os.path.join(csv_dir, f), delimiter=',',
        skip_header=False)  # np.loadtxt(os.path.join(csv_dir, f))
    # print(data.shape)
    if data.shape[0] == 64:
        csvs = np.concatenate((csvs, data), axis=1)
print(csvs.shape)
np.save('music-louder.npy', csvs)
