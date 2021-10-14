'''
Collect and pre-process the ground-truth/complete models into .h5 format, such that we don't need to process it every time for the instance correspondence.
'''

import os,sys,random
import pymeshlab as ml
import plyfile
import numpy as np 
import h5py

root_path = 'H:/RockScan'
rock_category = 'RR4'
num_folders = 36
output_path = "H:/Unity/Projects/Synthesis3D_test/SyntheticData/h5"
num_points_per_gt = 4096

for folderID in range(1,num_folders+1):
    fn = rock_category+'_'+str(folderID)+'.ply'
    print(f'Processing {fn}')
    model_path = os.path.join(root_path, rock_category, str(folderID), 'models')
    model_name = os.path.join(model_path, fn)
    ply = plyfile.PlyData().read(model_name)
    fields = np.array([list(x) for x in ply.elements[0]]).astype(np.float32)
    coords = np.ascontiguousarray(fields[:, :3] - fields[:, :3].mean(axis=0)) # shift to center at origin!

    # upsampling/downsampling to a fix point number
    # number of points per instance surface is usually less than 2048
    # to do this, we need to import mesh from array in pymeshlab, do surface reconstruction (ball pivoting), and poisson-disk sampling
    m = ml.Mesh(coords)
    ms = ml.MeshSet()
    ms.add_mesh(m, rock_category+'_'+str(folderID))
    ms.surface_reconstruction_ball_pivoting()
    ms.poisson_disk_sampling(samplenum=num_points_per_gt, exactnumflag=True)
    
    # re-sampling does not do exact number but close to it, so we pad or crop to obtain the exact number
    gt_points_resampled = ms.mesh(1).vertex_matrix() # mesh 0 is the original mesh, mesh 1 is the re-sampled mesh
    if (len(gt_points_resampled) > num_points_per_gt):
        # crop
        gt_points_resampled = gt_points_resampled[:num_points_per_gt,:]
    else:
        # pad
        padding = random.sample(list(range(len(gt_points_resampled))), num_points_per_gt - len(gt_points_resampled))
        selected_idx = list(range(len(gt_points_resampled))) + padding
        gt_points_resampled = gt_points_resampled[selected_idx, :]
    
    # write to h5
    save_fn = os.path.join(output_path, rock_category+'_'+str(folderID)+'.h5')
    with h5py.File(save_fn, 'w') as f:
        f.create_dataset('data', data=gt_points_resampled)
    # ms.save_current_mesh(os.path.join(output_path, fn))

    
    