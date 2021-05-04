import os
import pymeshlab as ml

# you can list all the available filters and their parameters
# ml.print_pymeshlab_version()
# ml.print_filter_list()
# ml.print_filter_parameter_list('surface_reconstruction_screened_poisson')

# output of filters (if any) is stored in dictionary
# out_dict = ms.compute_geometric_measures()
# print(out_dict['surface_area'])

root_path = 'H:\RockScan'
rock_category = 'RR3'
start_folder_ID = 1
end_folder_ID = 40
target_face_count = {0: 2000, 1: 1000, 2: 500} # at different LOD levels (LOD0 more faces --> LOD2 less faces) 

for folderID in range(start_folder_ID, end_folder_ID + 1): # folder name '1', '2', ... '40'
    model_path = os.path.join(root_path, rock_category, str(folderID), 'models')
    model_name = os.path.join(model_path, rock_category+'_'+str(folderID)+'.fbx')

    # meshset contains a set of meshes. Each mesh is a layer and has a unique ID.
    ms = ml.MeshSet()
    ms.load_new_mesh(model_name) # load new mesh and set as current mesh
    measures = ms.compute_geometric_measures()
    print("[PRINT] %s: %d vertices, %d faces, area %.1f cm^2, volume %.1f cm^3" % (rock_category+'_'+str(folderID)+'.fbx', ms.current_mesh().vertex_number(), ms.current_mesh().face_number(), measures['surface_area']*1e4, measures['mesh_volume']*1e6) )

    # Mesh Re-centering
    ms.transform_translate_center_set_origin(traslmethod='Center on Layer BBox', freeze=True)

    # Duplicate layers (for N LOD levels now we have N layers)
    for _ in range(len(target_face_count) - 1):
        ms.duplicate_current_layer()
    # print("Number of meshes: ", ms.number_meshes())  

    # Mesh Simplification
    for LOD, face_count in target_face_count.items():
        ms.set_current_mesh(LOD)
        ms.simplification_quadric_edge_collapse_decimation_with_texture(targetfacenum=face_count, preserveboundary=True, optimalplacement=True, preservenormal=True, planarquadric=True)
        #print("[PRINT] LOD%d: %d vertices, %d faces" % (LOD, ms.current_mesh().vertex_number(), ms.current_mesh().face_number()))

        # change working directory to where the model will be saved. This is important since the .obj and .mtl are linked by relative path. Must do this before export.
        os.chdir(model_path)
        save_model_name = rock_category+'_'+str(folderID)+'_'+'LOD'+str(LOD)+'.obj'
        ms.save_current_mesh(file_name=save_model_name)

# meshset object remembers every operation above, and we can save it and reuse it next time (or distribute to other users)
# ms.save_filter_script('my_script.mlx')
# reuse next time
# ms.load_new_mesh('another_input.obj')
# ms.load_filter_script('my_script.mlx')
# ms.apply_filter_script()
# ms.save_current_mesh('result.obj')