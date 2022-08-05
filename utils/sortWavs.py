import os
import shutil
import random

dryrun = False
shuffle = True
print("Dryrun:", dryrun)
print("Shuffle:", shuffle)

ORIGIN_FOLDER = "/mnt/backup3gb/datasets/voxceleb2-wav/wav"
TRAINING_FOLDER = "/home/my3bikaht/voxceleb2-wav-training"
TESTING_FOLDER = "/home/my3bikaht/voxceleb2-wav-testing"
speakersLimit = 100
trainingLimitPerSpeaker = 20
testingLimitPerSpeaker = 10

speakers = [
    name for name in os.listdir(ORIGIN_FOLDER)
    if os.path.isdir(os.path.join(ORIGIN_FOLDER, name))
]
speakers = speakers[:speakersLimit]

#create same directory structure
for speaker in speakers:
    if not dryrun:
        try:
            os.mkdir(TRAINING_FOLDER + os.sep + speaker)
            os.mkdir(TESTING_FOLDER + os.sep + speaker)
        except OSError as error:
            print(error)
    speakerFolder = ORIGIN_FOLDER + os.sep + speaker
    copyCandidates = []
    traningExamplesCopied = 0
    testingExamplesCopied = 0
    for currentPath, subdirs, filenames in os.walk(speakerFolder):
        #creating subfolders
        for subdir in subdirs:
            relativePath = currentPath.replace(speakerFolder, "")
            targetTrainingFolder = TRAINING_FOLDER + os.sep + speaker + relativePath + os.sep + subdir
            targetTestingFolder = TESTING_FOLDER + os.sep + speaker + relativePath + os.sep + subdir
            if dryrun:
                print("Creating training dir:", targetTrainingFolder)
                print("Creating testing dir:", targetTestingFolder)
            else:  # not dryrun
                try:
                    os.mkdir(targetTrainingFolder)
                except OSError as error:
                    print(error)
                try:
                    os.mkdir(targetTestingFolder)
                except OSError as error:
                    print(error)

        # copy files

        for filename in filenames:
            relativePath = currentPath.replace(speakerFolder, "")
            candidateFile = currentPath + '/' + filename
            extensions = ('.mp3', '.wav', '.mp4', ".ogg", '.flac', '.aac')
            if candidateFile.endswith(
                    extensions) and os.stat(candidateFile).st_size > 0:
                targetTrainingFile = TRAINING_FOLDER + os.sep + speaker + relativePath + os.sep + filename
                targetTestingFile = TESTING_FOLDER + os.sep + speaker + relativePath + os.sep + filename
                # to implement proper samples extraction, we need to put these
                # into archive and, if needed, shuffle

                # Generating list
                copyCandidates.append(
                    [candidateFile, targetTrainingFile, targetTestingFile])

    if shuffle:
        random.shuffle(copyCandidates)
    # Copying itself
    for copyCandidate in copyCandidates:
        if traningExamplesCopied < trainingLimitPerSpeaker:
            if dryrun:
                print("Copy traning example:", copyCandidate[0], ">>>",
                      copyCandidate[1])
            else:  # not dryrun
                try:
                    shutil.copy(copyCandidate[0], copyCandidate[1])
                except OSError as error:
                    print(error)
            traningExamplesCopied += 1
        else:
            if testingExamplesCopied < testingLimitPerSpeaker:
                if dryrun:
                    print("Copy testing example:", copyCandidate[0], ">>>",
                          copyCandidate[2])
                else:  # not dryrun
                    try:
                        shutil.copy(copyCandidate[0], copyCandidate[2])
                    except OSError as error:
                        print(error)
                testingExamplesCopied += 1
            else:
                #if dryrun:
                #    input("We hit copy limit per speaker>>>")
                break

    input(">>>>")
