import os
import numpy as np
import random
import torch.utils.data as D
import tqdm
import timeit


class Testset(D.Dataset):
    def __init__(self, dir1, dir2):
        list1 = os.listdir(dir1)
        list2 = os.listdir(dir2)
        self.list = [os.path.join(dir1, el) for el in list1
                     ] + [os.path.join(dir2, el) for el in list2]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        return np.load(self.list[index])


def load_test(dataloader):
    for batch in tqdm(dataloader):
        continue


def main():
    dir1 = '/mnt/nvme2tb/datasets/voxceleb2/double/cache/validate/0000000001'
    dir2 = '/media/sergey/386217cc-3490-42e9-b723-c3a32cc41f1f/tmp/0000000002'
    #dir2 = '/mnt/nvme2tb/datasets/voxceleb2/double/cache/validate/0000000002'
    dataset = Testset(dir1, dir2)
    dataloader = D.DataLoader(dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    starttime = timeit.default_timer()
    for batch in tqdm.tqdm(dataloader):
        continue
    print("Time taken:", timeit.default_timer() - starttime)


if __name__ == "__main__":
    main()
