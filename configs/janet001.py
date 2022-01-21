'''
WANDBPROJECTNAME = 'triplets-timit-simple-augmentation-test'
EPOCHS_TOTAL = 31
# GPU
GPU_ID = 0
# DATASETS & CACHE
NUM_CLASSES = 630
DATASET_TRAIN = "/mnt/nvme2tb/datasets/TIMIT2/simple/datasets/train.dt"
TRAIN_CACHE = "/mnt/nvme2tb/datasets/TIMIT2/simple/cache/train"
DATASET_VALIDATE = "/mnt/nvme2tb/datasets/TIMIT2/simple/datasets/validate.dt"
VALIDATE_CACHE = "/mnt/nvme2tb/datasets/TIMIT2/simple/cache/validate"
DATASET_TEST = None
TORCHINFO_SHAPE = (64, 1, 64, 192)  # shape or None
'''
# ------------------------- CONSTANTS -------------------------

# BASIC
WANDBPROJECTNAME = 'triplets-voxceleb2-double-pretrain'
EPOCHS_TOTAL = 9999
# GPU
GPU_ID = 0
# DATASETS & CACHE
NUM_CLASSES = 5994
DATASET_TRAIN = "/mnt/nvme2tb/datasets/voxceleb2/double/datasets/train.dt"
TRAIN_CACHE = ["/mnt/nvme2tb/datasets/voxceleb2/double/cache/train/"]
DATASET_VALIDATE = "/mnt/nvme2tb/datasets/voxceleb2/double/datasets/validate.dt"
VALIDATE_CACHE = ["/mnt/nvme2tb/datasets/voxceleb2/double/cache/validate/"]
TORCHINFO_SHAPE = (64, 1, 64, 192)  # shape or None

# MODEL
MODEL_LIBRARY_PATH = 'libs.models.janet_vox_v2'
MODEL_NAME = 'Janet'
# VISUALIZATION
VIZ_DIR = "./visualization/"
# CHECKPOINTS
RUN_ID = '1lpsocrw'
CHECKPOINTFILENAME = "./checkpoints/1lpsocrw064.dict"
# SUBSETS
USE_SUBSETS = True
SUBSET_SPEAKERS = 64
SUBSET_SPECTROGRAMS_PER_SPEAKER = 50
# TRIPLETS
MARGIN = .3
BATCH_SIZE = 32
TRIPLETSPERCLASS = 20
POSITIVECRITERION = 'Random'  # 'Random' vs 'Hard'.
# When POSITIVECRITERION == 'Hard' closest positive AND farthest
# negative will be selected not taking into account following
# negative strategies
NEGATIVESEMIHARD = 1  # select semi-hard negatives
NEGATIVEHARD = 1  # select hard negatives
AUGMENTATIONS = ['ambient_noise', 'gradclip']
AMBIENT_FILE = '/mnt/nvme2tb/datasets/ambient_noises/ambient.npy'
MUSIC_FILE = '/mnt/nvme2tb/datasets/music-genres-kaggle/music.npy'
