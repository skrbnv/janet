import os
import shutil
import random

SOURCE_DIR = "/mnt/nvme2tb/datasets/vctk/VCTK-Corpus/wav48"
TRAIN_DIR = "/mnt/nvme2tb/datasets/vctk/sorted/train"
VALIDATION_DIR = "/mnt/nvme2tb/datasets/vctk/sorted/validate"
TEST_DIR = "/mnt/nvme2tb/datasets/vctk/sorted/test"

folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]

for folder in folders:
    speaker = os.path.basename(folder)
    trainDir = TRAIN_DIR + '/' + speaker
    validateDir = VALIDATION_DIR + '/' + speaker
    testDir = TEST_DIR + '/' + speaker
    try:
        os.mkdir(trainDir)
    except Exception:
        continue
    try:
        os.mkdir(validateDir)
    except Exception:
        continue
    try:
        os.mkdir(testDir)
    except Exception:
        continue
    for root, dirnames, filenames in os.walk(folder):
        #sorted = [
        #    filename for filename in filenames if filename.endswith("mic1.flac")
        #]
        sorted = filenames

        random.shuffle(sorted)
        validateSet = sorted[:5]
        testSet = sorted[5:10]
        trainSet = sorted[10:]
        for filename in trainSet:
            shutil.copyfile(SOURCE_DIR + '/' + speaker + '/' + filename,
                            trainDir + '/' + filename)
        for filename in validateSet:
            shutil.copyfile(SOURCE_DIR + '/' + speaker + '/' + filename,
                            validateDir + '/' + filename)
        for filename in testSet:
            shutil.copyfile(SOURCE_DIR + '/' + speaker + '/' + filename,
                            testDir + '/' + filename)
