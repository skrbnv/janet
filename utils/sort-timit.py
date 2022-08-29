import os
import shutil
import random

SOURCE_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/combined"
TRAIN_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/sorted/train"
TEST_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/sorted/test"
TEST_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/sorted/test"

#groups = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
#for group in groups:
folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
for folder in folders:
    speaker = os.path.basename(folder)
    trainDir = TRAIN_DIR + '/' + speaker
    testDir = TEST_DIR + '/' + speaker
    testDir = TEST_DIR + '/' + speaker
    try:
        os.mkdir(trainDir)
    except Exception:
        continue
    try:
        os.mkdir(testDir)
    except Exception:
        continue
    try:
        os.mkdir(testDir)
    except Exception:
        continue
    for root, dirnames, filenames in os.walk(folder):
        sorted = [f for f in filenames if f.endswith(".wav")]

        random.shuffle(sorted)
        testSet = sorted[0:2]
        testSet = sorted[2:4]
        trainSet = sorted[4:]
        for filename in trainSet:
            shutil.copyfile(folder + '/' + filename,
                            trainDir + '/' + speaker + '_' + filename)
        for filename in testSet:
            shutil.copyfile(folder + '/' + filename,
                            testDir + '/' + speaker + '_' + filename)
        for filename in testSet:
            shutil.copyfile(folder + '/' + filename,
                            testDir + '/' + speaker + '_' + filename)
