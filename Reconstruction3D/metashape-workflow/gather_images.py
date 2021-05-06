'''
To prepare a more comprehensive training dataset for U2-Net, I need to gather the exported masks from Metashape, which is saved in [Rock Number]/masks folder.
'''

import os
import shutil
import glob

root_path = 'H:\RockScan'
rock_category = 'RR3'
num_folders = 40
u2net_path = 'H:\git_symphonylyh\3D\U-2-Net-Rock'
training_image_dir = os.path.join(u2net_path, 'train_data', 'ROCK', 'images'+os.sep)
training_label_dir = os.path.join(u2net_path, 'train_data', 'ROCK', 'masks'+os.sep)
image_ext = '.jpg'
label_ext = '.png'

for folder in range(1, num_folders + 1): 
    training_image_list = glob.glob(root_path + os.sep + rock_category + os.sep + str(folder) + os.sep + '*' + image_ext) # ./N/*.jpg
    training_label_list = glob.glob(root_path + os.sep + rock_category + os.sep + str(folder) + os.sep + 'masks' + os.sep + '*' + label_ext) # ./N/masks/*.png

    for src in training_image_list:
        shutil.copy(src, training_image_dir)
    for src in training_label_list:
        shutil.copy(src, training_label_dir)
