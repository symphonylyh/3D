'''
Gather the Level of Detail (LOD) models from Blender, which is saved in [Rock Number]/models folder.
'''

import os
import shutil

root_path = 'H:\RockScan'
rock_category = 'RR4'
start_folder_ID = 1
end_folder_ID = 36
dest_folder = 'H:\RockScan\LODs'

mesh_path = os.path.join(dest_folder, 'Mesh')
texture_path = os.path.join(dest_folder, 'Texture')
if not os.path.exists(mesh_path):
    os.makedirs(mesh_path)
if not os.path.exists(texture_path):
    os.makedirs(texture_path)

for folderID in range(start_folder_ID, end_folder_ID + 1): # folder name '1', '2', ... '40'
    model_path = os.path.join(root_path, rock_category, str(folderID), 'models')
    model_name = os.path.join(model_path, rock_category+'_'+str(folderID)+'_'+'LODs'+'.fbx')
    texture_name = os.path.join(model_path, rock_category+'_'+str(folderID)+'.jpg')
    
    shutil.copy(model_name, mesh_path)
    shutil.copy(texture_name, texture_path)
