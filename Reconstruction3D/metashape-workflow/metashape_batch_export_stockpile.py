'''
Usage: create an empty Metashape project, named "main.psx", Tools -- Run script

Use "%programfiles%\Agisoft\Metashape Pro\python\python.exe" -m pip install [python_module_name] in Windows CMD to install any package
'''
import Metashape, os
import numpy as np 
import pandas as pd 
import openpyxl

root_path = 'H:\RockScan'
rock_category = 'RR4'
num_folders = 36
project_name = 'reconstruct.psx'

print("Script started")
doc = Metashape.app.document
doc.clear()

volumes = np.zeros(num_folders)
areas = np.zeros(num_folders)

for i in range(1, num_folders + 1): # folder name '1', '2', ... '40'
    project_path = os.path.join(root_path, rock_category, str(i))
    doc.open(os.path.join(project_path, project_name))
    print("Processed project: " + project_path)

    merged_chunk = doc.chunks[2]
    volume = merged_chunk.model.volume() * 1e6 # cm^3
    area = merged_chunk.model.area() * 1e4 # cm^2
    volumes[i-1] = volume
    areas[i-1] = area
    print('Volume: ', volume, ' cm^3')
    print('Surface Area: ', area, ' cm^2')
    
    model_path = os.path.join(project_path, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_name = os.path.join(model_path, rock_category+'_'+str(i))
    merged_chunk.exportModel(path=model_name+'.fbx', binary=True, precision=6, texture_format=Metashape.ImageFormatJPEG, save_texture=True, save_uv=True, save_normals=True, save_colors=True, save_cameras=False, save_markers=False, save_udim =False, save_alpha=False, strip_extensions=False, format=Metashape.ModelFormatFBX)
    
    merged_chunk.exportModel(path=model_name+'.ply', clip_to_boundary = False, format = Metashape.ModelFormatPLY, binary=True, save_cameras = False, save_comment = False, save_markers = False, save_texture = False, save_uv = False, strip_extensions = True)
    print("Model saved.")

# write volume and surface area stats to excel
# Note: we should close Metashape app when we want to edit the excel file, otherwise there will be resouce conflict
summary_spreadsheet = os.path.join(root_path, rock_category+'.xlsx')

if os.path.exists(summary_spreadsheet):
    mode = 'a'
    writer = pd.ExcelWriter(summary_spreadsheet, mode=mode, if_sheet_exists='replace')
else:
    mode = 'w'
    writer = pd.ExcelWriter(summary_spreadsheet, mode=mode)
    
if mode == 'a':
    writer.book = openpyxl.load_workbook(summary_spreadsheet)
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
info = pd.DataFrame({
    'Rock ID': np.arange(1,num_folders+1),
    'Volume (cm^3)': volumes,
    'Surface Area (cm^2)': areas
})
info.to_excel(writer, sheet_name='Metashape Stats',float_format='%.2f', index=False)
writer.save()

print("Script finished.")