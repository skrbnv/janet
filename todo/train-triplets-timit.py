from torch import load as torch_load, device as torch_device
from torch.optim import SGD
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader
from numpy import mean as np_mean
import libs.functions as _fn
from libs.data import Dataset
import libs.models as models
import libs.losses as _losses
import libs.triplets as _tpl
#import libs.classifier as _cls
import libs.test as _test
import libs.training as _tr
from importlib import import_module
import wandb
import os
import torchinfo
import sys
import argparse

# ------------------------- MAIN -------------------------
_fn.report("**************************************************")
_fn.report("**               Training script                **")
_fn.report("**************************************************")
_fn.todolist()

# ------------------------- INIT -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--wandb',
                    action='store_true',
                    default=False,
                    help='sync with W&B')
parser.add_argument('--resume',
                    action='store_true',
                    default=False,
                    help='resume run')
parser.add_argument(
    '--config',
    action='store',
    default='configs.janet001-timit',
    help='config filename (including path) imported as module, \
        defaults to configs.default')
args = parser.parse_args()
RESUME, WANDB, cfg_path = args.resume, args.wandb, args.config

_fn.report(f'Importing configuration from \'{cfg_path}\'')
CFG = import_module(cfg_path)
RUN_ID = CFG.RUN_ID

# Generating unqiue hash for the training run
if not RESUME:
    RUN_ID = _fn.get_random_hash()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG.GPU_ID)

checkpoint = torch_load(CFG.CHECKPOINTFILENAME)

# Fixing seed for reproducibility
_fn.fix_seed(1)

if WANDB:
    if RESUME:
        print(
            f'Your run id is {RUN_ID} with checkpoint {CFG.CHECKPOINTFILENAME}'
        )
        input("Press any key if you want to continue >>")
        wprj = wandb.init(id=RUN_ID,
                          project=CFG.WANDBPROJECTNAME,
                          resume="must")
    else:
        wprj = wandb.init(project=CFG.WANDBPROJECTNAME, resume=False)
        RUN_ID = wprj.id

if cuda_is_available():
    device = torch_device("cuda:0")
else:
    device = torch_device("cpu")
_fn.report("Torch is using device:", device)

Model = getattr(models, CFG.MODEL_NAME)
model = Model(num_classes=CFG.NUM_CLASSES)

model.load_state_dict(checkpoint['state_dict'])
_fn.report("Model state dict loaded from checkpoint")

# overload classifier to embeddings (128 by default)
# and freeze all parameters for extractor block
del model.classifier
Classifier = getattr(models, CFG.TRIPLET_CLASSIFIER_NAME)
model.classifier = Classifier()
for param in model.extractor.parameters():
    param.requires_grad = False

model.float()
if cuda_is_available():
    model.cuda()
_fn.report(
    f'Model {CFG.MODEL_NAME} created, using \'{CFG.TRIPLET_CLASSIFIER_NAME}\' as classifier'
)

if CFG.TORCHINFO_SHAPE is not None:
    torchinfo.summary(model, CFG.TORCHINFO_SHAPE)

optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
_fn.report("Optimizer initialized")

#if RESUME:
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    _fn.report("Optimizer state dict loaded from checkpoint")

D = Dataset(filename=CFG.DATASET_TRAIN,
            cache_paths=CFG.CACHE_TRAIN,
            force_even=True)
D_eval = D.get_randomized_subset_with_augmentation(
    max_records=50,
    speakers_filter=D.get_unique_speakers(),
    augmentations_filter=[])
T = Dataset(filename=CFG.DATASET_TEST, cache_paths=CFG.CACHE_TEST)

test_loader = DataLoader(T, batch_size=32, shuffle=True)
train_eval_loader = DataLoader(D_eval, batch_size=32, shuffle=True)
_fn.report("Full train and test datasets loaded")

#ambient = NoiseLibrary(
#    CFG.AMBIENT_NOISE_FILE) if 'noise' in CFG.AUGMENTATIONS else None
#if ambient is not None:
#    _fn.report('Ambient noises library loaded')

########################################################
####          Setting up basic variables          ####
########################################################
initial_epoch = 0
if RESUME:
    initial_epoch = int(checkpoint['epoch']) + 1
    _fn.report("Initial epoch set to", initial_epoch)
    # finally release memory under checkpoint
    del checkpoint
criterion = _losses.CentroidLoss(CFG.NUM_CLASSES)

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

top1 = 0
#top1 = checkpoint['top1'] if RESUME and 'top1' in checkpoint.keys() else 0

lss = _losses.Losses()
for epoch in range(initial_epoch, CFG.EPOCHS_TOTAL):
    _fn.report("**************** Epoch", epoch, "out of", CFG.EPOCHS_TOTAL,
               "****************")

    # gc.collect()
    triplets_generated_per_epoch = 0
    _fn.report("-------------- Training ----------------")
    _fn.report(f'Generating subsets with {CFG.SUBSET_SPEAKERS} speakers')
    for batch_speakers in D.speakers_iterator(CFG.SUBSET_SPEAKERS, True):
        Dsub = D.get_randomized_subset(
            speakers_filter=batch_speakers,
            max_records=CFG.SUBSET_SPECTROGRAMS_PER_SPEAKER)
        train_loader = DataLoader(Dsub, batch_size=64, shuffle=True)

        _fn.report("Training cycle")
        losses = _tr.training_loop(train_loader,
                                   model=model,
                                   optimizer=optimizer,
                                   criterion=criterion)
        _fn.report(f'Loss: {np_mean(losses):.4f}')
        lss.append(losses, epoch)
    _fn.report(f'Epoch loss: {lss.mean(epoch):.4f}')
    """
    _fn.report("-------------- Visualization ----------------")
    if epoch > 0 and epoch % 1 == 0:
        dataset = _db.visualize(D, epoch, samples=30)
    """

    top1train, top5train, top1test, top5test = _test.test(model, D_eval, T)

    #centroids = D.calculate_centroids()
    #dmin, davg, dmax = _fn.centroid_distances(centroids)
    #_fn.report(
    #    f'Pairwise centroid distances, min:{dmin}, avg:{davg}, max:{dmax}')

    current_lr = optimizer.param_groups[0]['lr']

    if WANDB:
        wandb.log({
            "Loss": lss.mean(epoch),
            "Test loss": 0,
            "Top1 acc over training data": top1train,
            "Top5 acc over training data": top5train,
            "Top1 acc over test data": top1test,
            "Top5 acc over test data": top5test,
            #"Min pairwise distance": dmin,
            #"Avg pairwise distance": davg,
            #"Max pairwise distance": dmax,
            "Learning rate": current_lr,
            "Triplets generated": triplets_generated_per_epoch,
        })

    ##########################################################
    ##### Saving checkpoint if test accuracy improved
    ##########################################################
    if top1test > top1:
        top1 = top1test
        _fn.checkpoint(id=RUN_ID,
                       data={
                           'epoch': epoch,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           'top1': top1
                       })
    if _fn.early_stop(lss.mean_per_epoch(), criterion='min'):
        print("Early stop triggered")
        sys.exit(0)
