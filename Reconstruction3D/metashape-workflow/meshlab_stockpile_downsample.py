'''
The raw point cloud from Metashape is very dense, we downsample it to a relatively sparse cloud for segmentation network.
'''
import os, glob
import numpy as np 
import pymeshlab as ml 

root_path = 'H:/AggregateStockpile'
rock_category = 'RR3R'
start_folder = 1
end_folder = 3
dst_path = 'H:/AggregateStockpile/simplified_models_photo'
suffix = '_pcd.ply'

for folder in range(start_folder, end_folder + 1):
    model_path = os.path.join(root_path, rock_category, 'S'+str(folder), 'models/')
    target_model = glob.glob(model_path + '*' + suffix)[0]
    model_name = os.path.splitext(os.path.basename(target_model))[0]

    ms = ml.MeshSet()
    ms.load_new_mesh(target_model) # load new mesh and set as current mesh
    ms.point_cloud_simplification(samplenum=15000, exactnumflag=True) # for RR3, use 15000; for RR4 and RR3_RR4_Mix that occupies a larger area, use 30000 since there are more particles, we need to ensure number of points per particle (or maybe it's the per-real scale point density). For RR5K (kankakee) we use 40000

    ms.save_current_mesh(file_name=os.path.join(dst_path, model_name+'_simplified.ply'), binary=False, save_vertex_normal=False, save_vertex_color=True)
