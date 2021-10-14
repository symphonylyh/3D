'''
In the first attempt I forgot to shift the coordinates of each individual instances to center at the origin. To avoid repeating the long point sampling process, this is a patch file that reads the partial shapes and re-center them.
'''

import os, glob, h5py
import numpy as np

dataset_path = "H:/git_symphonylyh/3D/Completion3D/SnowflakeNet/datasets/rocks3d-rr3-rr4-mix/"
splits = ['train', 'test', 'val']
patch_subset = 'partial'

for split in splits:
    files = glob.glob(os.path.join(dataset_path, split, patch_subset, '001', '*.h5'))
    for file in files:
        with h5py.File(file, 'r+') as f:
            old_coords = np.array(f['data'])
            new_coords = old_coords - old_coords.mean(axis=0) # shift
            f['data'][...] = new_coords