# ------------------------- CONSTANTS -------------------------
# BASIC
WANDBPROJECTNAME = 'triplets-voxceleb2-double-pretrain'
EPOCHS_TOTAL = 9999
# GPU
GPU_ID = 0
# DATASETS & CACHE
NUM_CLASSES = 5994
DATASET_DIR = "/mnt/nvme2tb/datasets/voxceleb2/double/datasets/"
DATASET_TRAIN = "train.dt"
DATASET_VALIDATE = "validate.dt"
DATASET_TEST = "TIMIT-test.dt"
TRAIN_CACHE = "/mnt/nvme2tb/datasets/voxceleb2/double/cache/train/"
VALIDATE_CACHE = "/mnt/nvme2tb/datasets/voxceleb2/double/cache/validate/"
TORCHINFO_SHAPE = (64, 1, 64, 192)  # shape or None

# MODEL
MODEL_LIBRARY_PATH = 'libs.models.janet'
MODEL_NAME = 'Janet'
# VISUALIZATION
VIZ_DIR = "./visualization/"
# CHECKPOINTS
RUN_ID = '2qzvq9i4'
CHECKPOINTFILENAME = "./checkpoints/2qzvq9i4057.dict"
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
