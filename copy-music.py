import os
import shutil

mispredicted_list_file = './mispredicted.txt'
target_dir = './mispredicted'

with open(mispredicted_list_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        fname = line[25:].rstrip('\n')
        shutil.copyfile(fname, os.path.join(target_dir,
                                            os.path.basename(fname)))
