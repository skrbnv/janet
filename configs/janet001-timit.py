# ------------------------- CONSTANTS -------------------------

# BASIC
WANDBPROJECTNAME = 'triplets-timit-fastrun'
EPOCHS_TOTAL = 9999
# GPU
GPU_ID = 1
# DATASETS & CACHE
NUM_CLASSES = 630

DATASET_TRAIN = "/mnt/nvme2tb/datasets/TIMIT2/simple/datasets/train.dt"
CACHE_TRAIN = ["/mnt/nvme2tb/datasets/TIMIT2/simple/cache/train"]
DATASET_VALIDATE = "/mnt/nvme2tb/datasets/TIMIT2/simple/datasets/validate.dt"
CACHE_VALIDATE = ["/mnt/nvme2tb/datasets/TIMIT2/simple/cache/validate"]
TORCHINFO_SHAPE = (64, 1, 64, 192)  # shape or None

# MODEL
MODEL_NAME = 'Janet'
TRIPLET_CLASSIFIER_NAME = 'ClassifierEmbeddings'
# VISUALIZATION
VIZ_DIR = "./visualization/"
# CHECKPOINTS
RUN_ID = '1fbv3wj9'
CHECKPOINTFILENAME = "./checkpoints/27kii6em049.dict"
# SUBSETS
USE_SUBSETS = True
SUBSET_SPEAKERS = 64
SUBSET_SPECTROGRAMS_PER_SPEAKER = 50
# TRIPLETS
MARGIN = .3
BATCH_SIZE = 128
TRIPLETSPERCLASS = 20
POSITIVECRITERION = 'Random'  # 'Random' vs 'Hard'.
# When POSITIVECRITERION == 'Hard' closest positive AND farthest
# negative will be selected not taking into account following
# negative strategies
NEGATIVESEMIHARD = 1  # select semi-hard negatives
NEGATIVEHARD = 1  # select hard negatives
#AUGMENTATIONS = ['echo', 'gradclip']
#AMBIENT_NOISE_FILE = None
AUGMENTATIONS = ['mix', 'erase', 'label_smoothing'
                 ]  # mix, erase, noise, gradclip, label_smoothing
