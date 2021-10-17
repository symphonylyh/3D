'''
Generate the metadata json file for the dataset.
'''

import os 
import json 

dataset_name = 'rocks3d'

data = {}

data['taxonomy_id'] = "001"
data['taxonomy_name'] = "rock"

splits = ['train', 'test', 'val']

for split in splits:
    filestem = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(dataset_name, split, 'partial', '001'))]
    data[split] = filestem 

metadata = [data]
with open(dataset_name+'.json', 'w') as f:
    json.dump(metadata, f, indent=4)
