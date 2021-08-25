'''
Read laser scan STL file and convert to OBJ file with assigned texture.
'''
import os,sys
from shutil import copy 
import pymeshlab as ml
import numpy as np 
import pandas as pd 
import openpyxl

root_path = 'H:\RockScan'
rock_category = 'Ballast'
num_folders = 120
start_folder = 1
end_folder = 120

### Create 120 particle folders
# for i in range(1, num_folders+1):
#     folder_name = os.path.join(root_path, rock_category, str(i))
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#     if not os.path.exists(os.path.join(folder_name, 'models')):
#         os.makedirs(os.path.join(folder_name, 'models'))

### Move the laser scan .stl (mesh) & .txt (point cloud) model to the folder. The raw scan is 'gray_[01...40].stl', 'red_[01...40].stl', 'white_[01...40].stl'. Now gray rocks have index 1...40, red rocks have index 41...80, white rocks have index 81...120.
# stllist = [f for f in os.listdir(os.path.join(root_path, rock_category)) if os.path.isfile(os.path.join(root_path, rock_category, f)) and f.endswith('.stl')]
# print(stllist)

# for i in range(len(stllist)):
#     copy(os.path.join(root_path, rock_category, stllist[i]), os.path.join(root_path, rock_category, str(i+1), 'models', 'Ballast_'+str(i+1)+'.stl'))

### Copy the texture file from 1, 41, 81 to the other rocks
# for i in range(2, 41):
#     gray_texture = os.path.join(root_path, rock_category, str(1), 'models', 'Ballast_1.jpg')
#     copy(gray_texture, os.path.join(root_path, rock_category, str(i), 'models', 'Ballast_'+str(i)+'.jpg'))
# for i in range(42, 81):
#     red_texture = os.path.join(root_path, rock_category, str(41), 'models', 'Ballast_41.jpg')
#     copy(red_texture, os.path.join(root_path, rock_category, str(i), 'models', 'Ballast_'+str(i)+'.jpg'))
# for i in range(82, 121):
#     white_texture = os.path.join(root_path, rock_category, str(81), 'models', 'Ballast_81.jpg')
#     copy(white_texture, os.path.join(root_path, rock_category, str(i), 'models', 'Ballast_'+str(i)+'.jpg'))

### Smooth raw mesh from laser scanner
### Calculate volume and area
### Texture parameterization
### Set texture
### Save as .ply
volumes = np.zeros(num_folders)
areas = np.zeros(num_folders)
num_vertices = np.zeros(num_folders)
num_faces = np.zeros(num_folders)
for folderID in range(start_folder, end_folder+1):
    model_path = os.path.join(root_path, rock_category, str(folderID), 'models')
    model_name = os.path.join(model_path, rock_category+'_'+str(folderID)+'.stl')

    ms = ml.MeshSet()
    ms.load_new_mesh(model_name) # load new mesh and set as current mesh
    ms.remove_duplicate_vertices()
    ms.close_holes()

    # Taubin smooth, Filters -- Smoothing, Fairing, and Deformation -- Taubin smooth
    ms.taubin_smooth(stepsmoothnum=1)
    ms.re_orient_all_faces_coherentely()

    # UV parameterization, Filters -- Texture -- Parameterization: Flat Plane; Filters -- Texture -- Per vertex texture function; Filters -- Texture -- Set texture
    ms.per_vertex_texture_function() # assign UV coords # somehow, this step is required when we do in MeshLab, but in script we should remove this step. This generate per-vertex UV coords while the next line generates per-face texcoords. My guess is per-vertex overrides per-face when we open the saved mesh.
    
    ms.parametrization_flat_plane()
    ms.set_texture(textname=os.path.join(model_path, rock_category+'_'+str(folderID)+'.jpg'))

    # calculate volume and area
    measures = ms.compute_geometric_measures()
    print("[PRINT] %s: %d vertices, %d faces, area %.2f cm^2, volume %.2f cm^3" % (rock_category+'_'+str(folderID)+'.stl', ms.current_mesh().vertex_number(), ms.current_mesh().face_number(), measures['surface_area']/1e2, measures['mesh_volume']/1e3) )

    # laser scan unit is mm, convert to cm
    volumes[folderID-1] = measures['mesh_volume']/1e3
    areas[folderID-1] = measures['surface_area']/1e2
    num_vertices[folderID-1] = ms.current_mesh().vertex_number()
    num_faces[folderID-1] = ms.current_mesh().face_number()

    # texture to vertex color (otherwise Open3D won't be able to read the texture map)
    # ms.transfer_color_texture_to_vertex() # this doesn't work somehow...
    # this results in dark vertex color, we need to adjust the brightness
    # ms.vertex_color_brightness_contrast_gamma(brightness=0)

    # save ply model
    os.chdir(model_path)
    save_model_name = rock_category + '_' + str(folderID) + '.ply'
    ms.save_current_mesh(file_name=save_model_name, binary=False, save_vertex_normal=True, save_vertex_color=True, save_wedge_texcoord=True)
    save_model_name = rock_category + '_' + str(folderID) + '.obj'
    ms.save_current_mesh(file_name=save_model_name, save_vertex_normal=True, save_vertex_color=True, save_wedge_texcoord=True)
    # patch the missing texture in obj file when exported from Meshlab
    mat_name = save_model_name + '.mtl'
    texture_name = rock_category+'_'+str(folderID)+'.jpg'
    with open(mat_name, 'a') as f:
        f.write(f'map_Kd {texture_name}')


# write volume and surface area stats to excel
# summary_spreadsheet = os.path.join(root_path, rock_category+'.xlsx')

# if os.path.exists(summary_spreadsheet):
#     mode = 'a'
#     writer = pd.ExcelWriter(summary_spreadsheet, mode=mode, if_sheet_exists='replace')
# else:
#     mode = 'w'
#     writer = pd.ExcelWriter(summary_spreadsheet, mode=mode)
    
# if mode == 'a':
#     writer.book = openpyxl.load_workbook(summary_spreadsheet)
#     writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
# info = pd.DataFrame({
#     'Rock ID': np.arange(1,num_folders+1),
#     'Volume (cm^3)': volumes,
#     'Surface Area (cm^2)': areas,
#     'No. Vertices': num_vertices,
#     'No. Faces': num_faces
# })
# info.to_excel(writer, sheet_name='Metashape Stats',float_format='%.2f', index=False)
# writer.save()


