import random
import torch
import numpy as np
import gc
import libs.functions as _fn
from libs.data import Dataset
from libs.datasets import BasicDataset
import libs.losses as _losses
from libs.models.resnet_64x192_fast import Model
import libs.validation as _val
import libs.triplets as _tpl
import wandb
import os
import torchinfo
import sys
from libs.listen import Listen
# ------------------------- CONSTANTS -------------------------
# BASIC
WANDBPROJECTNAME = 'triplets-timit-simple'
EPOCHS_TOTAL = 9999
# GPU
GPU_ID = 1
# DATASETS & CACHE
DATASET_DIR = "/media/my3bikaht/EXT4/datasets/TIMIT2/datasets/"
DATASET_TRAIN = "glued-1_92s-mel64-plain-train.dt"
DATASET_VALIDATE = "glued-1_92s-mel64-plain-validate.dt"
DATASET_TEST = "TIMIT-test.dt"
SPECTROGRAMS_CACHE = "/media/my3bikaht/EXT4/datasets/TIMIT2/cache/glued-1_92s-mel64-plain"
# VISUALISATION
VIZ_DIR = "./visualization/"
# CHECKPOINTS
RUN_ID = '3hg1y70i'
CHECKPOINTFILENAME = "./checkpoints/3hg1y70i_e25.dict"
# SUBSETS
USE_SUBSETS = True
SUBSET_SPEAKERS = 64
SUBSET_SPECTROGRAMS_PER_SPEAKER = 50
# TRIPLETS
MARGIN = .3
BATCH_SIZE = 32
TRIPLETSPERCLASS = 20
POSITIVECRITERION = 'Random'
NEGATIVESEMIHARD = 0
NEGATIVEHARD = 0

# ------------------------- INIT -------------------------
args = [arg.replace('-', '').lower() for arg in sys.argv][1:]
RESUME = True if 'resume' in args else False
WANDB = True if 'wandb' in args else False
# Generating unqiue hash for the training run
__run_hash = RUN_ID if RESUME else _fn.get_random_hash()
listen = Listen()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

if RESUME:
    checkpoint = torch.load(CHECKPOINTFILENAME)

# Fixing seed for reproducibility
_fn.fix_seed(1)

if WANDB:
    if RESUME:
        print(f'Your run id is {RUN_ID} with checkpoint {CHECKPOINTFILENAME}')
        input("Press any key if you want to continue >>")
        wandb.init(id=RUN_ID, project=WANDBPROJECTNAME, resume="must")
    wandb.init(project=WANDBPROJECTNAME, resume=False)

# ------------------------- MAIN -------------------------
_fn.report("**************************************************")
_fn.report("**               Training script                **")
_fn.report("**************************************************")
_fn.todolist()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
_fn.report("Torch is using device:", device)

model = Model()
model.float()
if torch.cuda.is_available():
    model.cuda()
_fn.report("Model created")

torchinfo.summary(model, (64, 1, 64, 192))

if RESUME:
    model.load_state_dict(checkpoint['state_dict'])
    _fn.report("Model state dict loaded from checkpoint")

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-9)
_fn.report("Optimizer initialized")

if RESUME:
    optimizer.load_state_dict(checkpoint['optimizer'])
    _fn.report("Optimizer state dict loaded from checkpoint")

D = Dataset(filename=os.path.join(DATASET_DIR, DATASET_TRAIN),
            cache_path=SPECTROGRAMS_CACHE)
V = Dataset(filename=os.path.join(DATASET_DIR, DATASET_VALIDATE),
            cache_path=SPECTROGRAMS_CACHE)
_fn.report("Full train and validation datasets loaded")

########################################################
####          Setting up basic variables          ####
########################################################
initial_epoch = 0
if RESUME:
    initial_epoch = int(checkpoint['epoch']) + 1
    _fn.report("Initial epoch set to", initial_epoch)

criterion = torch.nn.MSELoss()

# Pre-generate triplet sets based on Spectogram class
# We have
# - Iterator over Speakers [.............]
# --- Generate set of anhchors by iterating over Samples * Spectograms
# --- Generate positives: spectograms belong to same speaker except anchor
# --- Generate negatives: all spectograms that don't belong to same Speaker

########################################################
####                  NN cycle                      ####
########################################################
if WANDB:
    wandb.watch(model)
bestT1 = 0
p = _fn.Status()

speakers = D.get_speakers_subset(256)
Dsub = D.get_randomized_subset(speakers, max_records=1000)
params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
trainingSet = BasicDataset(Dsub)
trainingGenerator = torch.utils.data.DataLoader(trainingSet, **params)
lent = Dsub.total()

for epoch in range(initial_epoch, EPOCHS_TOTAL):
    _fn.report("**************** Epoch", epoch, "out of", EPOCHS_TOTAL,
               "****************")

    losses = []
    _fn.report("-------------- Training ----------------")
    for i, (b_spg, b_speaker) in enumerate(trainingGenerator):
        optimizer.zero_grad()
        y_pred = model(b_spg.to(device))
        loss = criterion(y_pred, b_speaker.float().to(device))
        loss.backward()
        p.print(f'Loss: {loss.item():.4f}, {i*32} out of {lent}')
        optimizer.step()
        losses.append(loss.item())
    _fn.report(f'Epoch loss: {np.mean(losses):.4f}')
    """
    _fn.report("-------------- Visualization ----------------")
    if epoch > 0 and epoch % 1 == 0:
        dataset = _db.visualize(D, epoch, samples=30)
    """
    '''
    top1train, top5train, top1val, top5val = _val.validate(model, D, V, device)

    centroids = D.calculate_centroids()
    dmin, davg, dmax = _fn.centroid_distances(centroids)
    _fn.report(
        f'Pairwise centroid distances, min:{dmin}, avg:{davg}, max:{dmax}')

    current_lr = optimizer.param_groups[0]['lr']

    if WANDB:
        wandb.log({
            "Loss": np.mean(losses),
            "Top1 acc over training data": top1train,
            "Top5 acc over training data": top5train,
            "Top1 acc over validation data": top1val,
            "Top5 acc over validation data": top5val,
            "Min pairwise distance": dmin,
            "Avg pairwise distance": davg,
            "Max pairwise distance": dmax,
            "Learning rate": current_lr,
            "Triplets generated": 0
        })
    '''

    ##########################################################
    ##### Saving checkpoint if validation accuracy improved
    ##########################################################
    chkptfname = f'./checkpoints/{__run_hash}_e{epoch}.dict'
    torch.save(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, chkptfname)
    _fn.report(f'Checkpoint {chkptfname} saved')
