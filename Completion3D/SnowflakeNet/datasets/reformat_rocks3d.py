'''
Reformat the rocks3d dataset.
'''

import os, shutil

dataset_name = 'rocks3d'

filenames = [os.path.basename(f) for f in os.listdir(os.path.join(dataset_name, 'train', 'partial', '001'))]

for f in filenames:
    fields = f.split('_')
    cat, rock_id = fields[1], fields[2]
    if cat == 'RR3' and int(rock_id) > 40:
        # partial
        src = os.path.join(dataset_name, 'train', 'partial', '001', f)
        dst = os.path.join(dataset_name, 'test', 'partial', '001', f)
        shutil.move(src, dst)

        # gt
        src = os.path.join(dataset_name, 'train', 'gt', '001', f)
        dst = os.path.join(dataset_name, 'test', 'gt', '001', f)
        shutil.move(src, dst)