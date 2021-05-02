# Usage: create an empty Metashape project, named "main.psx"
import Metashape, os

root_path = 'H:\RockScan'
rock_category = 'RR3'
num_folders = 40
project_name = 'reconstruct.psx'

print("Script started")
doc = Metashape.app.document
doc.clear()

for i in range(1, num_folders + 1): # folder name '1', '2', ... '40'
    project_path = os.path.join(root_path, rock_category, str(i))
    doc.open(os.path.join(project_path, project_name))
    print("Processed project: " + project_path)

    merged_chunk = doc.chunks[2]
    print('Volume: ', merged_chunk.model.volume() * 1e6, ' cm^3')
    
    model_path = os.path.join(project_path, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_name = os.path.join(model_path, rock_category+'_'+str(i)+'.fbx')
    merged_chunk.exportModel(path=model_name, binary=True, precision=6, texture_format=Metashape.ImageFormatJPEG, save_texture=True, save_uv=True, save_normals=True, save_colors=True, save_cameras=False, save_markers=False, save_udim =False, save_alpha=False, strip_extensions=False, format=Metashape.ModelFormatFBX)
    print("Model saved.")


print("Script finished.")