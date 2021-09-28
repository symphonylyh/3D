# call by: python blender_entry.py
# blender script for merging LOD models into .fbx package
# Tested on Blender 2.92

import os
import bpy

root_path = 'H:\RockScan'
rock_category = 'RR4'
start_folder_ID = 1
end_folder_ID = 36
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

# set scene unit (important, otherwise Unity won't set the mesh collider correctly)
# scale = 0.01 means 1 Blender unit = 0.01m = 1cm
# FBX system uses 1 unit = 1cm by default, so here we set the same and export
C.scene.unit_settings.system = 'METRIC'
C.scene.unit_settings.scale_length = 0.01

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
    bpy.ops.export_scene.fbx(filepath=saved_model_name, global_scale=1.00)

    # clean all objects
    for obj in D.objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    # clean all materials
    for mat in D.materials:
        D.materials.remove(mat)
