'''
Example for visualizing Primitives3D and Rock3D ply data.
Usage:
    - python plot3d_example_synthetic3d.py
    - modify dataset_path, file_id, draw_all, block_range, sem_dict
This script plots 3 figures:
    - Fig 1: Raw point cloud & Colored by block
    - Fig 2: Colored by semantic & Colored by instance
    - Fig 3: Colored by instance with bbox
By switching the draw_all flag, you can choose to draw the entire scene or just certain blocks
'''

import os, sys
import glob
import plyfile
import numpy as np

from plot3d import Plot3DApp, Plot3DFigure, PointCloudVis as pcvis

###  Read point cloud file
dataset_path = '/home/luojiayi/Documents/haohang/3D/Segmentation3D/PointGroup/dataset/primitives3d/ply/'
file_id = 0
sem_dict = ['cube', 'sphere', 'cylinder'] # or sem_dict = {0:'cube', 1:'sphere', 2:'cylinder'}

files = sorted(glob.glob(dataset_path + '*.ply'))
print(f"Dataset has {len(files)} stockpiles, plotting stockpile {file_id}")

f = plyfile.PlyData().read(files[file_id])
points = np.array([list(x) for x in f.elements[0]]).astype(np.float32)
coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(axis=0))
colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
sem_labels = np.ascontiguousarray(points[:, 7])
ins_labels = np.ascontiguousarray(points[:, 8])

# expand dims to fit the plot3d (B,N,X) structure
coords = coords[np.newaxis,:,:]
colors = colors[np.newaxis,:,:]
sem_labels = sem_labels[np.newaxis,:,np.newaxis]
ins_labels = ins_labels[np.newaxis,:,np.newaxis]

### Init plot function
app = Plot3DApp()

fig1 = app.create_figure(figure_name='Fig 1', viewports_dim=(1,1), width=1280, height=720, sync_camera=True, plot_boundary=True, show_axes=True, show_subtitles=True, background_color=(1,1,1,1), snapshot_path='./')

fig2 = app.create_figure(figure_name='Fig 2', viewports_dim=(1,2), width=1280, height=720, sync_camera=True, plot_boundary=False, show_axes=True, show_subtitles=True, background_color=(1,1,1,1), snapshot_path='./')

fig3 = app.create_figure(figure_name='Fig 3', viewports_dim=(1,1), width=1280, height=720, sync_camera=True, plot_boundary=False, show_axes=True, show_subtitles=True, background_color=(1,1,1,1), snapshot_path='./')

### Plot visualization to different figures
pc_xyzrgb = np.concatenate( (coords[:,:,:], colors[:,:,:]), axis=2) # B * N * 6
pc_xyzrgbsemins = np.concatenate( (pc_xyzrgb, sem_labels[:,:,:], ins_labels[:,:,:]), axis=2) # B * N * 8


fig1a = fig1.set_subplot(0,0,'Raw Point Cloud')
pcvis.draw_pc_raw(fig1a, pc_xyzrgb)

fig2a = fig2.set_subplot(0,0,'Point Cloud by Semantic')
pcvis.draw_pc_by_semins(fig2a, pc_xyzrgbsemins, sem_dict=sem_dict, color_code='semantic', show_legend=True)
fig2b = fig2.set_subplot(0,1,'Point Cloud by Instance')
pcvis.draw_pc_by_semins(fig2b, pc_xyzrgbsemins, sem_dict=None, color_code='instance', show_bbox=True)

fig3a = fig3.set_subplot(0,0,'Point Cloud by Instance')
pcvis.draw_pc_by_semins(fig3a, pc_xyzrgbsemins, sem_dict=sem_dict, color_code='instance', show_bbox=True, bbox_axis_align=True, bbox_color='black', show_instance_label=True)

### Start GUI
fig1.ready() # must call ready!
fig2.ready()
fig3.ready()

app.plot()
app.close()
