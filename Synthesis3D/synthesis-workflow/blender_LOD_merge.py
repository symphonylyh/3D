# call by: python blender_entry.py
# blender script for merging LOD models into .fbx package
# Tested on Blender 2.92

import os
import bpy

root_path = 'H:\RockScan'
rock_category = 'RR3'
start_folder_ID = 1
end_folder_ID = 40
LODs = 3

C = bpy.context
D = bpy.data 
# clean all objects
for obj in D.objects:
    obj.select_set(True)
bpy.ops.object.delete()
# clean all materials
for mat in D.materials:
    D.materials.remove(mat)

for folderID in range(start_folder_ID, end_folder_ID + 1): # folder name '1', '2', ... '40'
    model_path = os.path.join(root_path, rock_category, str(folderID), 'models')
    
    for LOD in range(LODs):
        model_name = os.path.join(model_path, rock_category+'_'+str(folderID)+'_'+'LOD'+str(LOD)+'.obj')
        bpy.ops.import_scene.obj(filepath=model_name) # https://docs.blender.org/api/current/bpy.ops.import_scene.html#bpy.ops.import_scene.obj

    # for obj in D.objects:
    #     print(obj.name, obj.type)

    # let different LOD levels share the same material
    material = D.objects[0].data.materials[0]
    for obj in D.objects:
        obj.data.materials[0] = material

    # save .fbx model
    saved_model_name = os.path.join(model_path, rock_category+'_'+str(folderID)+'_'+'LODs'+'.fbx')
    bpy.ops.export_scene.fbx(filepath=saved_model_name)

    # clean all objects
    for obj in D.objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    # clean all materials
    for mat in D.materials:
        D.materials.remove(mat)
