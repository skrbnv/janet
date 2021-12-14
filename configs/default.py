# ------------------------- CONSTANTS -------------------------
# BASIC
WANDBPROJECTNAME = 'triplets-timit-simple'
EPOCHS_TOTAL = 9999
# GPU
GPU_ID = 0
# DATASETS & CACHE
DATASET_DIR = "/mnt/nvme2tb/datasets/TIMIT2/datasets/"
DATASET_TRAIN = "glued-1_92s-mel64-plain-train.dt"
DATASET_VALIDATE = "glued-1_92s-mel64-plain-validate.dt"
DATASET_TEST = "TIMIT-test.dt"
SPECTROGRAMS_CACHE = "/mnt/nvme2tb/datasets/TIMIT2/cache/glued-1_92s-mel64-plain"
TORCHINFO_SHAPE = (64, 1, 64, 192)  # shape or None
# MODEL
MODEL_LIBRARY_PATH = 'libs.models.resnet_models'
MODEL_NAME = 'ResNet10_64x192_kernel5'
# VISUALISATION
VIZ_DIR = "./visualization/"
# CHECKPOINTS
RUN_ID = '2jypfazr'
CHECKPOINTFILENAME = "./checkpoints/2jypfazr00.dict"
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
