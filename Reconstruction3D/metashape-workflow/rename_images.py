'''
On Windows, when I import photos, everytime it will start numbering at 001.JPG. The rock images are separated into sub-folders for reconstruction, but I will also aggregate all the images with masks into the training dataset for U2-Net, so there should not be duplicate names. This script is to rename images 001.JPG, 002.JPG, ... --> N.JPG, N+1.JPG, ... from a given starting N.
'''

import os

folder_name = 'H:\RockScan\RR3'
file_ext = '.JPG'
start_index = 2385 # rock 32

l = [f for f in os.listdir(folder_name) if f.endswith(file_ext)]
lsorted = sorted(l,key=lambda x: int(os.path.splitext(x)[0])) # just to make sure the images are correctly ordered by the number

for i,f in enumerate(lsorted):
    os.rename(f, str(start_index+i)+file_ext)
