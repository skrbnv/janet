'''
WANDBPROJECTNAME = 'triplets-timit-simple-augmentation-test'
EPOCHS_TOTAL = 31
# GPU
GPU_ID = 0
# DATASETS & CACHE
NUM_CLASSES = 630
DATASET_TRAIN = "/mnt/nvme2tb/datasets/TIMIT2/simple/datasets/train.dt"
CACHE_TRAIN = "/mnt/nvme2tb/datasets/TIMIT2/simple/cache/train"
DATASET_VALIDATE = "/mnt/nvme2tb/datasets/TIMIT2/simple/datasets/validate.dt"
CACHE_VALIDATE = "/mnt/nvme2tb/datasets/TIMIT2/simple/cache/validate"
DATASET_TEST = None
TORCHINFO_SHAPE = (64, 1, 64, 192)  # shape or None
'''
# ------------------------- CONSTANTS -------------------------

# BASIC
WANDBPROJECTNAME = 'triplets-voxceleb2-fastrun'
EPOCHS_TOTAL = 9999
# GPU
GPU_ID = 1
# DATASETS & CACHE
NUM_CLASSES = 5994

DATASET_TRAIN = "/mnt/nvme2tb/datasets/voxceleb2/fastrun/filtered-nocompressor/train.dt"
CACHE_TRAIN = [
    "/mnt/nvme2tb/datasets/voxceleb2/fastrun/filtered-nocompressor/cache/train",
    "/mnt/nvme1tb/datasets/voxceleb2/fastrun/filtered-nocompressor/cache/train"
]
DATASET_VALIDATE = "/mnt/nvme2tb/datasets/voxceleb2/fastrun/filtered-nocompressor/validate.dt"
CACHE_VALIDATE = [
    "/mnt/nvme2tb/datasets/voxceleb2/fastrun/filtered-nocompressor/cache/validate",
    "/mnt/nvme1tb/datasets/voxceleb2/fastrun/filtered-nocompressor/cache/validate"
]
TORCHINFO_SHAPE = (64, 1, 64, 192)  # shape or None

# MODEL
MODEL_LIBRARY_PATH = 'libs.models.convnext_v1'
MODEL_NAME = 'ConvNeXt'
# VISUALIZATION
VIZ_DIR = "./visualization/"
# CHECKPOINTS
RUN_ID = '16byjx1b'
CHECKPOINTFILENAME = "./checkpoints/16byjx1b006.dict"
# SUBSETS
USE_SUBSETS = True
SUBSET_SPEAKERS = 64
SUBSET_SPECTROGRAMS_PER_SPEAKER = 50
# TRIPLETS
MARGIN = .3
BATCH_SIZE = 64
TRIPLETSPERCLASS = 20
POSITIVECRITERION = 'Random'  # 'Random' vs 'Hard'.
# When POSITIVECRITERION == 'Hard' closest positive AND farthest
# negative will be selected not taking into account following
# negative strategies
NEGATIVESEMIHARD = 1  # select semi-hard negatives
NEGATIVEHARD = 1  # select hard negatives
#AUGMENTATIONS = ['echo', 'gradclip']
#AMBIENT_NOISE_FILE = None
AUGMENTATIONS = ['mix', 'erase', 'noise',
                 'gradclip']  # mix, erase, noise, gradclip
AMBIENT_NOISE_FILE = '/mnt/nvme2tb/datasets/voxceleb2/spectrogram_noises/noises.npy'
