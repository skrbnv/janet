from libs.data import merge_cache

PRIMARY_CACHE = '/mnt/nvme2tb/datasets/voxceleb2/double/cache'
AUXILLARY_CACHE = '/mnt/nvme2tb/datasets/voxceleb2/double/cache-validate'

merge_cache(PRIMARY_CACHE, AUXILLARY_CACHE)
