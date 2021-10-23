import os
import shutil
import random

SOURCE_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/combined"
TRAIN_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/sorted/train"
VALIDATION_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/sorted/validate"
TEST_DIR = "/home/my3bikaht/datasets/timit/TIMIT2/sorted/test"

#groups = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
#for group in groups:
folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
for folder in folders:
	speaker = os.path.basename(folder)
	trainDir = TRAIN_DIR + '/' + speaker
	validateDir = VALIDATION_DIR + '/' + speaker
	testDir = TEST_DIR + '/' + speaker
	try:
		os.mkdir(trainDir)
	except:
		continue
	try:
		os.mkdir(validateDir)
	except:
		continue
	try:
		os.mkdir(testDir)
	except:
		continue
	for root, dirnames, filenames in os.walk(folder):
		sorted = [f for f in filenames if f.endswith(".wav")]

		random.shuffle(sorted)
		validateSet = sorted[0:2]
		testSet = sorted[2:4]
		trainSet = sorted[4:]
		for filename in trainSet:
			shutil.copyfile(folder + '/' + filename,
							trainDir + '/' + speaker + '_' + filename)
		for filename in validateSet:
			shutil.copyfile(folder + '/' + filename,
							validateDir + '/' + speaker + '_' + filename)
		for filename in testSet:
			shutil.copyfile(folder + '/' + filename,
							testDir + '/' + speaker + '_' + filename)
