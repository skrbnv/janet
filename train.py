import torch
import libs.functions as _fn
from libs.data import Dataset
import libs.losses as _losses
import libs.classifier as _cls
import libs.models as models
import wandb
import os
import torchinfo
import argparse
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # ------------------------- MAIN -------------------------
    _fn.report("**************************************************")
    _fn.report("**             Pre-training script              **")
    _fn.report("**  Pretraining model using cross entropy loss  **")
    _fn.report("**           instead of triplet loss.           **")
    _fn.report("**************************************************")
    _fn.todolist()
    _fn.fix_seed(42)
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
    parser.add_argument('--notrain',
                        action='store_true',
                        default=False,
                        help='Skip training altogether')
    parser.add_argument(
        '--config',
        action='store',
        default='timit',
        help='config filename (including path) imported as module, \
            defaults to configs.default')
    args = parser.parse_args()
    RESUME, WANDB, FREEZE, NOTRAIN, cfg = args.resume, args.wandb, args.freeze, args.notrain, args.config

    CONFIG = _fn.load_yaml(cfg)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        CONFIG['general']['gpu_id']['value'])

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
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _fn.report("Torch is using device:", device)

    Model = getattr(models, CONFIG['model']['name']['value'])
    model = Model(num_classes=CONFIG['general']['classes']['value'])

    model.float()
    model.to(device)
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
    torchinfo.summary(model,
                      tuple(CONFIG['general']['torchinfo_shape']['value']))

    # Setting up initial epoch
    initial_epoch = 0
    if RESUME:
        initial_epoch = int(checkpoint['epoch']) + 1
        _fn.report("Initial epoch set to", initial_epoch)

    # Setting up criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Setting up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.SGD(
    #    model.parameters(),
    #    lr=CONFIG['optimizer']['initial_lr']['value'],
    #    momentum=CONFIG['optimizer']['momentum']['value'],
    #    weight_decay=CONFIG['optimizer']['weight_decay']['value'],
    #    nesterov=CONFIG['optimizer']['nesterov']['value'])
    #if RESUME:
    #    optimizer.load_state_dict(checkpoint['optimizer'])
    #    _fn.report("Optimizer state dict loaded from checkpoint")
    _fn.report("Optimizer initialized")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=200)
    #scheduler = StepDownScheduler(optimizer,
    #                              initial_epoch=initial_epoch,
    #                              config=CONFIG['scheduler'])
    #_fn.report("Scheduler initialized")

    # Setting up datasets and data loaders
    D = Dataset(filename=CONFIG['dataset']['train']['file']['value'],
                cache_paths=CONFIG['dataset']['train']['dirs']['value'],
                force_even=True,
                useindex=CONFIG['dataset']['useindex']['value'],
                caching=False)
    DT = D.get_randomized_subset_with_augmentation(
        max_records=50,
        speakers_filter=D.get_unique_speakers(),
        augmentations_filter=[],
        useindex=CONFIG['dataset']['useindex']['value'],
        caching=False)
    T = Dataset(filename=CONFIG['dataset']['test']['file']['value'],
                cache_paths=CONFIG['dataset']['test']['dirs']['value'],
                useindex=CONFIG['dataset']['useindex']['value'],
                caching=False)

    train_loader = DataLoader(
        D,
        batch_size=CONFIG['general']['batch_size']['value'],
        shuffle=True,
        num_workers=0)
    test_loader = DataLoader(
        T,
        batch_size=CONFIG['general']['batch_size']['value'],
        shuffle=False,
        num_workers=0)
    train_eval_loader = DataLoader(
        DT,
        batch_size=CONFIG['general']['batch_size']['value'],
        shuffle=False,
        num_workers=0)
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

        if not NOTRAIN:
            losses = _cls.train(
                train_loader,
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

        top1train, top5train, top1test, top5test, test_loss = _cls.test(
            train_eval_loader, test_loader, model, criterion)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"T1train: {top1train}, T5train: {top5train}, T1test: {top1test}, T5test: {top5test}, lr: {current_lr:.4f}"
        )

        if WANDB:
            wandb.log({
                "Loss": lss.mean(epoch),
                "Test loss": test_loss,
                "Top1 acc over training data": top1train,
                "Top5 acc over training data": top5train,
                "Top1 acc over test data": top1test,
                "Top5 acc over test data": top5test,
                "Learning rate": current_lr
            })

        ##########################################################
        ##### Saving checkpoint if testing accuracy improved
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
        #if _fn.early_stop(lss.mean_per_epoch(), criterion='min'):
        #    print("Early stop triggered")
        #    sys.exit(0)
        scheduler.step()
