'''
Example for visualizing the synthetic generated raw PLY data (Unity data) with LiDAR view.
Usage:
    - python plot3d_example_ply.py
    - modify ply_path, file_id, sem_dict
This script plots 2 figures:
    - Fig 1: Raw point cloud (shifted to zero origin)
    - Fig 2: Point cloud color-coded by LiDAR ID (shifted to zero origin)
This script is intended to illustrate the ray tracing process of LiDAR when generating the synthetic data. This script is separated from plot3d_example_synthetic3d.py since the LiDAR position information is not carried to h5 training data.
'''

import os, sys, glob
from plyfile import PlyData, PlyElement
import numpy as np 

from plot3d import Plot3DApp, Plot3DFigure, PointCloudVis as pcvis

ply_path = 'H:/Unity/Projects/Synthesis3D_test/SyntheticData/pointcloud/'

start_fid = 90
end_fid = 90

for file_id in range(start_fid, end_fid+1):
    ply_files = sorted(glob.glob(ply_path + '*.ply'))
    f = ply_files[file_id]
    print(f)

    plydata = PlyData.read(f)

    points = plydata['vertex'].data # this is numpy structured array, i.e. with fields
    # print(plydata['vertex'].properties) # show all fields

    (x, y, z, r, g, b, lidar_id, sem_id, ins_id) = (points[field] for field in ('x', 'y', 'z', 'diffuse_red', 'diffuse_green', 'diffuse_blue', 'lidar_ID', 'semantic_ID', 'instance_ID')) 
    pc_xyzrgb = np.stack((x,y,z,r,g,b), axis=1)
    pc_xyzrgbsemins = np.stack((x,y,z,r,g,b, sem_id, ins_id), axis=1)

    # Parse comments to extract LiDAR positions
    comments = plydata.comments
    lidar_positions = []
    for c in comments:
        tokens = c.split()
        if tokens[0] == 'LiDAR':
            pos = [float(tokens[x]) for x in (-3,-2,-1)]
            lidar_positions.append(pos)
    lidar_positions = np.asarray(lidar_positions)

    # Prep 1: shift coords and lidar positions start from (0,0,0)
    origin = np.mean(pc_xyzrgb[:,0:3], axis=0)
    pc_xyzrgb[:,0:3] -= origin 
    pc_xyzrgbsemins[:,0:3] -= origin 
    lidar_positions -= origin

    # Prep 2: force color in [0,1] scale
    if np.max(pc_xyzrgb[:, 3:6]) > 1: 
        pc_xyzrgb[:, 3:6] /= 255.0

    ins_labels = pc_xyzrgbsemins[:,-1].flatten()
    ins_keys = np.unique(ins_labels)
    ins_keys = ins_keys[ins_keys > -1]
    print(f"Total: {len(ins_keys)} instances")

    # Visualize data
    app = Plot3DApp()

    fig1 = app.create_figure(figure_name='Fig1', viewports_dim=(1,2), width=1280, height=720, sync_camera=True, plot_boundary=False, show_axes=False, show_subtitles=True, background_color=(1,1,1,2), snapshot_prefix='raycasting', snapshot_path='./')
    fig1a = fig1.set_subplot(0,0,'Raw Point Cloud')
    pcvis.draw_pc_raw(fig1a, pc_xyzrgb)
    fig1b = fig1.set_subplot(0,1,'Point Cloud by Instance Label')
    pcvis.draw_pc_by_semins(fig1b, pc_xyzrgbsemins[np.newaxis, :,:], line_width=1.5)

    fig2 = app.create_figure(figure_name='Fig2', viewports_dim=(1,1), width=1280, height=720, sync_camera=True, plot_boundary=False, show_axes=False, show_subtitles=True, background_color=(1,1,1,2), snapshot_prefix='raycasting', snapshot_path='./')
    fig2a = fig2.set_subplot(0,0,'Point Cloud by LiDAR')
    pcvis.draw_pc_by_lidar(fig2a, np.concatenate((pc_xyzrgb, lidar_id.reshape(-1,1)), axis=1), lidar_positions, show_rays=True, show_rays_lidar=[0,13])

    fig1.ready()
    fig2.ready()
    app.plot()
    app.close()

