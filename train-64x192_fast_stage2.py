import random
import torch
import numpy as np
import gc
import libs.functions as _fn
from libs.data import Dataset
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
RUN_ID = '25ev09ki'
CHECKPOINTFILENAME = "./checkpoints/64x192_fast_trained.dict"
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
_fn.report("Optimizer initialized")

#if RESUME:
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    _fn.report("Optimizer state dict loaded from checkpoint")

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

criterion = _losses.CustomTripletMarginLoss(MARGIN)

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
for epoch in range(initial_epoch, EPOCHS_TOTAL):
    _fn.report("**************** Epoch", epoch, "out of", EPOCHS_TOTAL,
               "****************")

    # gc.collect()

    losses_all = []
    triplets_generated_per_epoch = 0
    _fn.report("-------------- Training ----------------")
    _fn.report(f'Generating subsets with {SUBSET_SPEAKERS} speakers')
    for batch_speakers in D.speakers_iterator(SUBSET_SPEAKERS, True):
        Dsub = D.get_randomized_subset(
            speakers_filter=batch_speakers,
            max_records=SUBSET_SPECTROGRAMS_PER_SPEAKER)

        _fn.report("Updating embeddings")
        model.eval()
        Dsub.update_embeddings(model=model.innerModel, device=device)
        model.train()

        _fn.report("Generating triplets")
        Dsub.reset()
        anchors, positives, negatives = _tpl.generate_triplets_mp(
            Dsub, TRIPLETSPERCLASS, POSITIVECRITERION, NEGATIVESEMIHARD,
            NEGATIVEHARD)

        _fn.report(f'Triplets generated: {len(anchors)}')
        triplets_generated_per_epoch += len(anchors)

        _fn.report("Training cycle")
        model_params = {
            'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }
        losses = _tpl.train_triplet_model_dual(Dsub,
                                               model=model,
                                               params=model_params,
                                               optimizer=optimizer,
                                               anchors=anchors,
                                               positives=positives,
                                               negatives=negatives,
                                               epoch=epoch,
                                               device=device,
                                               criterion=criterion)
        _fn.report(f'Loss: {np.mean(losses):.4f}')
        losses_all.extend(losses)
    _fn.report(f'Epoch loss: {np.mean(losses_all):.4f}')
    """
    _fn.report("-------------- Visualization ----------------")
    if epoch > 0 and epoch % 1 == 0:
        dataset = _db.visualize(D, epoch, samples=30)
    """

    if epoch % 10 == 0 or epoch == initial_epoch:
        top1train, top5train, top1val, top5val = _val.validate(
            model, D, V, device)

    centroids = D.calculate_centroids()
    dmin, davg, dmax = _fn.centroid_distances(centroids)
    _fn.report(
        f'Pairwise centroid distances, min:{dmin}, avg:{davg}, max:{dmax}')

    current_lr = optimizer.param_groups[0]['lr']

    if WANDB:
        wandb.log({
            "Loss": np.mean(losses_all),
            "Top1 acc over training data": top1train,
            "Top5 acc over training data": top5train,
            "Top1 acc over validation data": top1val,
            "Top5 acc over validation data": top5val,
            "Min pairwise distance": dmin,
            "Avg pairwise distance": davg,
            "Max pairwise distance": dmax,
            "Learning rate": current_lr,
            "Triplets generated": triplets_generated_per_epoch
        })

    ##########################################################
    ##### Saving checkpoint if validation accuracy improved
    ##########################################################
    if epoch > 0 and top1val > bestT1:
        bestT1 = top1val
        chkptfname = "./checkpoints/" + __run_hash + "_epoch_" + str(
            epoch).zfill(2) + ".dict"
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, chkptfname)
        _fn.report(f'Checkpoint {chkptfname} saved')
    if listen.check('exit'):
        sys.exit(1)
    nlr = listen.check('lr')
    if nlr:
        for pg in optimizer.param_groups:
            pg['lr'] = nlr
        _fn.report(f'Learning rate updated to {nlr}')
