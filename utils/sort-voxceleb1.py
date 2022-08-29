### Sorting script for voxceleb2 dataset
### (sorting into train and test parts)
# Records belonging to the same speaker randomly
# distributed between training and test datasets
# with some probability (.8 / .2 by default)

import os
import shutil
import random

SOURCE_DIR = "/media/my3bikaht/EXT4/datasets/voxceleb1/_original_sources/vox1_dev_wav/wav"
TRAIN_DIR = "/media/my3bikaht/EXT4/datasets/voxceleb1/sorted/train"
TEST_DIR = "/media/my3bikaht/EXT4/datasets/voxceleb1/sorted/test"
TEST_CHANCE = .01
MAX_TO_TEST = 10
MIN_TO_TEST = 5

folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]

for i, folder in enumerate(folders):
    print("\r Parsing %d out of %d speakers" % (i, len(folders)), end='')

    speaker = os.path.basename(folder)
    sourceDir = SOURCE_DIR + '/' + speaker
    trainDir = TRAIN_DIR + '/' + speaker
    testDir = TEST_DIR + '/' + speaker
    try:
        os.mkdir(trainDir)
    except Exception:
        continue
    try:
        os.mkdir(testDir)
    except Exception:
        continue
    source_list = []
    for speakerDir, sampleNames, _ in os.walk(folder):
        for sampleName in sampleNames:
            for root, subdirs, filenames in os.walk(speakerDir + "/" +
                                                    sampleName):
                for filename in filenames:
                    source_list.append({
                        'speaker': speaker,
                        'sample': sampleName,
                        'file': filename
                    })
    # pick random records to test
    to_test_count = int(TEST_CHANCE * len(source_list))
    if to_test_count < MIN_TO_TEST:
        to_test_count = MIN_TO_TEST
    if to_test_count > MAX_TO_TEST:
        to_test_count = MAX_TO_TEST
    random.shuffle(source_list)
    to_val = source_list[0:to_test_count]
    to_train = source_list[to_test_count:]
    for record in to_val:
        shutil.copyfile(
            sourceDir + '/' + record['sample'] + '/' + record['file'],
            testDir + '/' + record['sample'] + '_' + record['file'])
    for record in to_train:
        shutil.copyfile(
            sourceDir + '/' + record['sample'] + '/' + record['file'],
            trainDir + '/' + record['sample'] + '_' + record['file'])
