'''
From synthetic stockpile PLY file:
    - Extract points of each instance (partial point cloud)
    - Based on the prototype ID, parse and fetch the corresponding full model (ground-truth or complete point cloud)
    - Save as the right format (.h5) & Organize in folder. The taxnomy ID we use is just a random one, `001`. We first convert all instances in the train folder, then move to test/val folders.
'''
import os,sys,glob,random
from shutil import copy 
import pymeshlab as ml
import plyfile
import numpy as np 
import h5py
import open3d as o3d 
import uuid

category_map = {
    3: 'RR3',
    4: 'RR4'
}

stockpile_path = "H:/Unity/Projects/Synthesis3D_test/SyntheticData/RR4_Completion"
prototype_path = "H:/Unity/Projects/Synthesis3D_test/SyntheticData/h5"
output_path = "H:/git_symphonylyh/3D/Completion3D/SnowflakeNet/datasets/rocks3d-rr4/train"
min_ins_points = 300 # minimum number of points to be an instance
num_points_per_ins = 2048

filelist = [f for f in os.listdir(stockpile_path) if f.endswith(".ply")]

for f in filelist:
    fn = os.path.join(stockpile_path, f)
    print(f'Processing {f}')
    ply = plyfile.PlyData().read(fn)
    fields = np.array([list(x) for x in ply.elements[0]]).astype(np.float32)
    # (x,y,z,r,g,b,lidar_id, semamtic_id, instance_id, prototype_id)

    coords = np.ascontiguousarray(fields[:, :3])
    # colors = np.ascontiguousarray(fields[:, 3:6]) / 127.5 - 1 # from [0-255] to [-1,1]
    ins_ids = np.ascontiguousarray(fields[:, 8]).astype(np.int32)
    proto_ids = np.ascontiguousarray(fields[:, 9]).astype(np.int32)

    unique_instances = np.unique(ins_ids)
    num_valid_instances = 0
    for ins_id in unique_instances:
        ins_points = coords[ins_ids == ins_id,:]
        proto_id = np.unique(proto_ids[ins_ids == ins_id])
        assert len(proto_id) == 1, "All instance points should correspond to the same prototype"

        # only consider instance having enough points
        if len(ins_points) < min_ins_points:
            continue
        num_valid_instances += 1

        # upsampling/downsampling to a fix point number
        # number of points per instance surface is usually less than 2048
        # to do this, we need to import mesh from array in pymeshlab, do surface reconstruction (ball pivoting), and poisson-disk sampling
        m = ml.Mesh(ins_points)
        ms = ml.MeshSet()
        ms.add_mesh(m, str(ins_id))
        ms.surface_reconstruction_ball_pivoting()
        ms.poisson_disk_sampling(samplenum=num_points_per_ins, exactnumflag=True)
        
        # re-sampling does not do exact number but close to it, so we pad or crop to obtain the exact number
        ins_points_resampled = ms.mesh(1).vertex_matrix() # mesh 0 is the original mesh, mesh 1 is the re-sampled mesh
        if (len(ins_points_resampled) > num_points_per_ins):
            # crop
            ins_points_resampled = ins_points_resampled[:num_points_per_ins,:]
        else:
            # pad
            padding = random.sample(list(range(len(ins_points_resampled))), num_points_per_ins - len(ins_points_resampled))
            selected_idx = list(range(len(ins_points_resampled))) + padding
            ins_points_resampled = ins_points_resampled[selected_idx, :]
        
        # [IMPORTANT step] center the resampled points at origin! Previously we shift the entire stockpile, however, that makes no difference, we actually need to shift EACH instance to its own origin for the network to efficiently learn. Same idea applies to the complete prototype model, they need to be at the same origin & the same scale (the physics simulation should be 1:1)
        ins_points_resampled = ins_points_resampled - ins_points_resampled.mean(axis=0) # shift to start at origin
        # scale to range [0,1] in each axis
        # average_scale = np.sum(ins_points_resamples.max(axis=0)) / 3 
        # ins_points_resampled /= average_scale

        # find the prototype model     
        cat, id = proto_id[0] // 100, proto_id[0] % 100
        prototype = category_map[cat] + '_' + str(id)
        
        # random filename
        save_fn = uuid.uuid4().hex 
        partial_path = os.path.join(output_path, 'partial', save_fn+'.h5')
        gt_path = os.path.join(output_path, 'gt', save_fn+'.h5')

        # save partial clouds; copy & rename gt clouds
        with h5py.File(partial_path, 'w') as f:
            f.create_dataset('data', data=ins_points_resampled)
        copy(os.path.join(prototype_path, prototype+'.h5'), gt_path)
    
    print(f'{fields.shape[0]} points, {num_valid_instances}/{len(unique_instances)} valid instances')
        