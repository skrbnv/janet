import os
import pickle


class CachedWrites():
    ''' Implements saving spectrograms in temp big files as a set of 1000 elements '''
    def __init__(self, save_dir, filename_prefix='file') -> None:
        self.cache = {}
        self.path = save_dir
        self.fprefix = filename_prefix

    def write(self, obj, filename, force=False):
        if len(self.cache) >= 1000 or force is True:
            num = len(os.listdir(self.path))
            fname = os.path.join(self.path, f'{self.fprefix}{num:04d}.obj')
            with open(fname, 'wb') as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Batch saved to {fname}')
            self.cache = {}
        else:
            pass
        if obj is not None:
            self.cache[filename] = obj

    def finalize(self):
        self.write(None, None, True)
