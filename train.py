import torch
#import numpy as np
#import gc
import libs.functions as _fn
from libs.data import Dataset
import libs.losses as _losses
from libs.scheduler import StepDownScheduler
#import libs.validation as _val
import libs.classifier as _cls
import libs.models as models
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
_fn.fix_seed(1)
# ------------------------- INIT -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--wandb',
                    action='store_true',
                    default=True,
                    help='sync with W&B')
parser.add_argument('--resume',
                    action='store_true',
                    default=True,
                    help='resume run')
parser.add_argument('--freeze',
                    action='store_true',
                    default=False,
                    help='freeze extractor layers')
parser.add_argument(
    '--config',
    action='store',
    default='vox2',
    help='config filename (including path) imported as module, \
        defaults to configs.default')
args = parser.parse_args()
RESUME, WANDB, FREEZE, cfg = args.resume, args.wandb, args.freeze, args.config

CONFIG = _fn.load_yaml(cfg)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG['general']['gpu_id']['value'])

if RESUME:
    checkpoint = torch.load(CONFIG['general']['checkpoint']['value'])

if WANDB:
    if RESUME:
        checkpoint_file = CONFIG['general']['checkpoint']['value']
        RUN_ID = os.path.basename(checkpoint_file).rstrip('.dict')[:-3]
        print(f'Your run id is {RUN_ID} with checkpoint {checkpoint_file}')
        input("Press any key if you want to continue >>")
        wprj = wandb.init(id=RUN_ID,
                          project=CONFIG['wandb']['project']['value'],
                          resume="must",
                          config=CONFIG)
    else:
        wprj = wandb.init(project=CONFIG['wandb']['project']['value'],
                          resume=False,
                          config=CONFIG)
        RUN_ID = wprj.id
else:
    RUN_ID = _fn.get_random_hash()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
_fn.report("Torch is using device:", device)

Model = getattr(models, CONFIG['model']['name']['value'])
model = Model(num_classes=CONFIG['general']['classes']['value'])

model.float()
if torch.cuda.is_available():
    model.cuda()
_fn.report(f"Model {CONFIG['model']['name']['value']} created")

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

print(model)
torchinfo.summary(model, tuple(CONFIG['general']['torchinfo_shape']['value']))

# Setting up initial epoch
initial_epoch = 0
if RESUME:
    initial_epoch = int(checkpoint['epoch']) + 1
    _fn.report("Initial epoch set to", initial_epoch)

# Setting up criterion
criterion = torch.nn.CrossEntropyLoss()

# Setting up optimizer and scheduler
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=CONFIG['optimizer']['initial_lr']['value'],
    momentum=CONFIG['optimizer']['momentum']['value'],
    weight_decay=CONFIG['optimizer']['weight_decay']['value'],
    nesterov=CONFIG['optimizer']['nesterov']['value'])
#if RESUME:
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    _fn.report("Optimizer state dict loaded from checkpoint")
_fn.report("Optimizer initialized")

scheduler = StepDownScheduler(optimizer,
                              initial_epoch=initial_epoch,
                              config=CONFIG['sheduler'])
#_fn.report("Scheduler initialized")

# Setting up datasets and data loaders
D = Dataset(filename=CONFIG['dataset']['train']['file']['value'],
            cache_paths=CONFIG['dataset']['train']['dirs']['value'],
            force_even=True)
DT = D.get_randomized_subset_with_augmentation(
    max_records=50,
    speakers_filter=D.get_unique_speakers(),
    augmentations_filter=[])
V = Dataset(filename=CONFIG['dataset']['valid']['file']['value'],
            cache_paths=CONFIG['dataset']['valid']['dirs']['value'])

train_loader = DataLoader(D, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(V, batch_size=32, shuffle=False, num_workers=0)
train_eval_loader = DataLoader(DT, batch_size=32, shuffle=True, num_workers=0)
_fn.report("Datasets loaded")

########################################################
####                  NN cycle                      ####
########################################################
if WANDB:
    wandb.watch(model)

top1 = checkpoint['top1'] if RESUME and 'top1' in checkpoint.keys() else 0
lss = _losses.Losses()
for epoch in range(initial_epoch, CONFIG['general']['epochs']['value']):
    _fn.report("**************** Epoch", epoch, "out of",
               CONFIG['general']['epochs']['value'], "****************")
    lss = _losses.Losses()
    _fn.report("-------------- Training ----------------")

    losses = _cls.train(train_loader,
                        model,
                        optimizer,
                        criterion,
                        augmentations=CONFIG['augmentations']['value'],
                        num_classes=CONFIG['general']['classes']['value'],
                        extras={})
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
    #if _fn.early_stop(lss.mean_per_epoch(), criterion='min'):
    #    print("Early stop triggered")
    #    sys.exit(0)
    #scheduler.step()
