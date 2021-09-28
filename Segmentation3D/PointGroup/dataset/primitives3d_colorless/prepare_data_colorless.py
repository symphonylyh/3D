'''
Convert from ply to pth for pytorch dataloader.

Usage:
    - place this file in the dataset folder, at the same level with /train, /val, /test splits
    - python prepare_data_colorless.py --data_split train # or val, test
    - val is the validation set having ground-truth semins labels, test is the data in production that has no ground-truth labels (only xyzrgb)

Notes:
    - different from pointgroup's prepare_data.py, our data is double, we need to convert to float since pointgroup cuda operation requires torch float tensor, so we need cast to np.float32 during data pre-processing.
    - color/colorless data are treated differently, this file is for colorless data pre-processing. colors are forced to zero in trainval/test loader.
'''

import glob, plyfile, numpy as np, multiprocessing as mp, torch, argparse

def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]]).astype(np.float32)
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(axis=0))
    colors = np.zeros_like(points[:, 3:6]) # remove color info

    torch.save((coords, colors), fn[:-4] + '.pth')
    print('Saving to ' + fn[:-4] + '.pth')

def f_trainval(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]]).astype(np.float32) # pointgroup uses float tensor, so cast to float
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(axis=0))
    colors = np.zeros_like(points[:, 3:6]) # remove color info

    sem_labels = np.ascontiguousarray(points[:, 7])
    ins_labels = np.ascontiguousarray(points[:, 8])
    
    torch.save((coords, colors, sem_labels, ins_labels), fn[:-4]+'.pth')
    print('Saving to ' + fn[:-4]+'.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', help='data split (train / val / test)', default='train')

    opt = parser.parse_args()
    split = opt.data_split
    print('data split: {}'.format(split))

    files = sorted(glob.glob(split + '/*.ply'))

    p = mp.Pool(processes=mp.cpu_count())
    if opt.data_split == 'test':
        p.map(f_test, files)
    else:
        p.map(f_trainval, files)
    p.close()
    p.join()
