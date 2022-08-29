from libs.data import Dataset, cache_validate
import tqdm

DATASET = '/mnt/nvme2tb/datasets/voxceleb2/double/datasets/test.dt'
CACHE = "/mnt/nvme2tb/datasets/voxceleb2/double/cache-validate"

D = Dataset(filename=DATASET, cache_path=CACHE)

for i in tqdm.tqdm(range(len(D.data))):
    if not cache_validate(D.data[i]['cacheId'], CACHE):
        raise Exception(f'Spectrogram {D.data[i]["cacheId"]} missing in cache')
