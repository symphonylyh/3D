'''
Usage: create an empty Metashape project, named "main.psx", Tools -- Run script

Use "%programfiles%\Agisoft\Metashape Pro\python\python.exe" -m pip install [python_module_name] in Windows CMD to install any package
'''
import Metashape, os
import numpy as np 

root_path = 'd:\AggregateStockpile_Kankakee'
rock_category = 'RR5K'
num_folders = 3
project_name = 'reconstruct.psx'

print("Script started")
doc = Metashape.app.document
doc.clear()

for i in range(1, num_folders + 1): # folder name 'S1', 'S2', ... 'SN'
    project_path = os.path.join(root_path, rock_category, 'S'+str(i))
    doc.open(os.path.join(project_path, project_name))
    print("Processed project: " + project_path)

    source_type = ['photo', 'frame'] # each project has two chunks, one reconstructed with photos, one with video frames

    for j, source in enumerate(source_type):
        chunk = doc.chunks[j]

        model_path = os.path.join(project_path, 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_name = os.path.join(model_path, rock_category+'_'+'S'+str(i)+'_'+source)
        
        chunk.exportModel(path=model_name+'_mesh.ply', clip_to_boundary = False, format = Metashape.ModelFormatPLY, binary=True, save_cameras = False, save_comment = False, save_markers = False, save_texture = False, save_uv = False, strip_extensions = True)

        chunk.exportPoints(path=model_name+'_pcd.ply', source_data=Metashape.DenseCloudData, binary=True, save_normals=True, save_colors=True, save_classes=False, save_confidence=False, save_comment = False, clip_to_boundary = False) # (x,y,z,nx,ny,nz,r,g,b) rgb is in 0-255 range

    print("Point cloud and mesh models saved.")

print("Script finished.")