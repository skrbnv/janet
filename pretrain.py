import torch
from configs.janet001 import CACHE_TRAIN
#import numpy as np
#import gc
import libs.functions as _fn
from libs.data import Dataset, NoiseLibrary
import libs.losses as _losses
#import libs.validation as _val
import libs.classifier as _cls
from importlib import import_module
import wandb
import os
import torchinfo
import sys
import argparse
from torch.utils.data import DataLoader

# ------------------------- MAIN -------------------------
_fn.report("**************************************************")
_fn.report("**             Pre-training script              **")
_fn.report("**  Pretraining model using cross entropy loss  **")
_fn.report("**           instead of triplet loss.           **")
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
parser.add_argument('--freeze',
                    action='store_true',
                    default=False,
                    help='freeze extractor layers')
parser.add_argument('--memcache',
                    action='store_true',
                    default=False,
                    help='Cache data to memory, at least partially')
parser.add_argument(
    '--config',
    action='store',
    default='configs.janet001',
    help='config filename (including path) imported as module, \
        defaults to configs.default')
args = parser.parse_args()
RESUME, WANDB, FREEZE, MEMCACHE, cfg_path = args.resume, args.wandb, args.freeze, args.memcache, args.config

_fn.report(f'Importing configuration from \'{cfg_path}\'')
CFG = import_module(cfg_path)
RUN_ID = CFG.RUN_ID

# Generating unqiue hash for the training run
if not RESUME:
    RUN_ID = _fn.get_random_hash()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG.GPU_ID)

if RESUME:
    checkpoint = torch.load(CFG.CHECKPOINTFILENAME)

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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
_fn.report("Torch is using device:", device)

Model = getattr(import_module(CFG.MODEL_LIBRARY_PATH), CFG.MODEL_NAME)
model = Model()

model.float()
if torch.cuda.is_available():
    model.cuda()
_fn.report(f"Model {CFG.MODEL_NAME} from {CFG.MODEL_LIBRARY_PATH} created")

if RESUME:
    model.load_state_dict(checkpoint['state_dict'])
    _fn.report("Model state dict loaded from checkpoint")
if FREEZE:
    if not RESUME:
        raise Exception('Process started anew, cannot freeze new layers')
    if model.extractor:
        for param in model.extractor.parameters():
            param.requires_grad = False
        _fn.report('Model Extractor block parameters are frozen')
    else:
        raise Exception('No \'extractor\' block in model')

if CFG.TORCHINFO_SHAPE is not None:
    torchinfo.summary(model, CFG.TORCHINFO_SHAPE)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
_fn.report("Optimizer initialized")

if RESUME:
    optimizer.load_state_dict(checkpoint['optimizer'])
    _fn.report("Optimizer state dict loaded from checkpoint")

D = Dataset(filename=CFG.DATASET_TRAIN,
            cache_paths=CFG.CACHE_TRAIN,
            force_even=True,
            use_memcache=MEMCACHE)
D_eval = D.get_randomized_subset_with_augmentation(
    max_records=50,
    speakers_filter=D.get_unique_speakers(),
    augmentations_filter=[0],
    use_memcache=MEMCACHE)
V = Dataset(filename=CFG.DATASET_VALIDATE,
            cache_paths=CFG.CACHE_VALIDATE,
            use_memcache=MEMCACHE)

train_workers = len(CFG.CACHE_TRAIN) if isinstance(CFG.CACHE_TRAIN,
                                                   list) else 0
train_loader = DataLoader(D,
                          batch_size=32,
                          shuffle=True,
                          num_workers=train_workers)
valid_workers = len(CFG.CACHE_VALIDATE) if isinstance(CFG.CACHE_VALIDATE,
                                                      list) else 0
valid_loader = DataLoader(V,
                          batch_size=32,
                          shuffle=True,
                          num_workers=valid_workers)
train_eval_loader = DataLoader(D_eval,
                               batch_size=32,
                               shuffle=True,
                               num_workers=train_workers)
_fn.report("Full train and validation datasets loaded")

ambient = NoiseLibrary(
    CFG.AMBIENT_NOISE_FILE) if 'noise' in CFG.AUGMENTATIONS else None
if ambient is not None:
    _fn.report('Ambient noises library loaded')

########################################################
####          Setting up basic variables          ####
########################################################
initial_epoch = 0
if RESUME:
    initial_epoch = int(checkpoint['epoch']) + 1
    _fn.report("Initial epoch set to", initial_epoch)

criterion = torch.nn.CrossEntropyLoss()

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
    lss = _losses.Losses()
    _fn.report("-------------- Training ----------------")

    losses = _cls.train(train_loader,
                        model,
                        optimizer,
                        criterion,
                        augmentations=CFG.AUGMENTATIONS,
                        num_classes=CFG.NUM_CLASSES,
                        extras={'ambient': ambient})
    lss.append(losses, epoch)
    _fn.report(f'Epoch loss: {lss.mean(epoch):.4f}')
    """
    _fn.report("-------------- Visualization ----------------")
    if epoch > 0 and epoch % 1 == 0:
        dataset = _db.visualize(D, epoch, samples=30)
    """

    top1train, top5train, top1val, top5val, val_loss = _cls.validate(
        train_eval_loader, valid_loader, model, criterion)

    print(
        f"T1T: {top1train}, T5T: {top5train}, T1V: {top1val}, T5V: {top5val}")

    current_lr = optimizer.param_groups[0]['lr']

    if WANDB:
        wandb.log({
            "Loss": lss.mean(epoch),
            "Validation loss": val_loss,
            "Top1 acc over training data": top1train,
            "Top5 acc over training data": top5train,
            "Top1 acc over validation data": top1val,
            "Top5 acc over validation data": top5val,
            "Learning rate": current_lr
        })

    ##########################################################
    ##### Saving checkpoint if validation accuracy improved
    ##########################################################
    if top1val > top1:
        top1 = top1val
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
