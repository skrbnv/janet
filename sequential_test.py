import torch
import libs.functions as _fn
import librosa
import os
import libs.models as models
import pprint

#checkpointfilename = '/home/sergey/code/cleanup/checkpoints/1jupj2ou001.dict'
checkpointfilename = '/home/sergey/code/cleanup/checkpoints/3dvwk62l051.dict'
test_dir = '/mnt/nvme2tb/datasets/TIMIT2/sorted/test'
config = {
    'SLICE_MS': 1920,
    'STEP_MS': 100,
    'TRIM_MS': 100,
    'MAX_SPECTROGRAMS_PER_SAMPLE': 100,
    'SLICING_STRATEGY': 'glue',
    'SKIPSHORTSLICES': .5,
    'MEL_BANKS': 64,
    'FORCE_SHAPE': False,
    'FORCE_SHAPE_SIZE': (224, 224)
}

MODEL_NAME = 'Janet'
TRIPLET_CLASSIFIER_NAME = 'ClassifierEmbeddings'
NUM_CLASSES = 630

_fn.report("**************************************************")
_fn.report("**            Sequential test script            **")
_fn.report("**       Tesing sequences of spectrograms       **")
_fn.report("**             for each audio sample.           **")
_fn.report("**************************************************")

spk_indices = torch.load('/stuff/speaker_indices_timit')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
_fn.report("Torch is using device:", device)

Model = getattr(models, MODEL_NAME)
model = Model(num_classes=NUM_CLASSES)
_fn.report("Model initialized")
checkpoint = torch.load(checkpointfilename)
_fn.report("Checkpoint loaded")
model.load_state_dict(checkpoint['state_dict'])
_fn.report("Model state dict loaded from checkpoint")
if torch.cuda.is_available():
    model.cuda()
model.eval()

fnames = []
# Get all speakers / subfolders' names from test dir
folders = [f.path for f in os.scandir(test_dir) if f.is_dir()]
for folder in folders:
    files = [f.path for f in os.scandir(folder) if f.name.endswith('.wav')]
    for file in files:
        fnames.append([os.path.basename(folder), file])

results = []
with torch.no_grad():
    for speaker, fname in fnames:
        audio, sr = librosa.load(fname, sr=16000, mono=True)
        spgs, rms = _fn.generate_slices(audio,
                                        length=config['SLICE_MS'] / 1000,
                                        sr=16000,
                                        step=config['STEP_MS'] / 1000,
                                        trim=config['TRIM_MS'] / 1000,
                                        strategy=config['SLICING_STRATEGY'],
                                        min_len=config['SKIPSHORTSLICES'],
                                        n_mels=config['MEL_BANKS'],
                                        augm=0,
                                        config=config)

        if spgs is None:
            raise Exception('No spectrograms generated')
        preds = []
        for spg in spgs:
            spectrogram = spg[0] + 40
            spectrogram /= 40
            pred = model(
                torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(
                    0).to(device))
            preds.append(torch.argmax(pred).item())
            #print(pred)
        results.append([speaker, spk_indices[speaker], preds])

correct = 0
total = 0
for speaker, index, preds in results:
    predictions = {}
    pred_index, pred_count = None, None
    upreds = list(set(preds))
    for upred in upreds:
        count = preds.count(upred)
        predictions[upred] = count
        if pred_count is None or pred_count < count:
            pred_index = upred
            pred_count = count

    if index == pred_index:
        correct += 1
    else:
        print(
            f'{speaker}[{index}] - {pred_index}/{pred_count}/{(pred_count*100/len(preds)):.2f}%'
        )
        pprint.pprint(predictions)

    total += 1

print(f'Sequential predictions accuracy: {(correct*100/total):.2f}%')
