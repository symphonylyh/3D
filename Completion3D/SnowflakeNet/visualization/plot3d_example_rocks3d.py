'''
Example for visualizing Rock3D shape completion data.
Usage:
    - python plot3d_example_rocks3d.py
    - modify dataset_path, file_id, draw_all, block_range, sem_dict
This script plots 3 figures:
    - Fig 1: Raw point cloud & Colored by block
    - Fig 2: Colored by semantic & Colored by instance
    - Fig 3: Colored by instance with bbox
By switching the draw_all flag, you can choose to draw the entire scene or just certain blocks
Note:
    - somehow on Ubuntu the bbox cannot be plotted in Open3D
'''

import os, sys
import h5py
import numpy as np

from plot3d import Plot3DApp, Plot3DFigure, PointCloudVis as pcvis

###  Read point cloud file
dataset_path_partial = '../datasets/rocks3d-rr3-rr4-mix/test/partial/001/'
dataset_path_gt = '../datasets/rocks3d-rr3-rr4-mix/test/gt/001/'
file_id = 0
dataset_path_inference = '../datasets/rocks3d-rr3-rr4-mix/test/benchmark/001/'

files = [os.path.basename(f) for f in os.listdir(dataset_path_partial)]
fn = files[file_id]
print(f"Dataset has {len(files)} samples, plotting sample {file_id}: {fn}")

filepath_partial = os.path.join(dataset_path_partial, fn)
filepath_gt = os.path.join(dataset_path_gt, fn)

filepath_pcd0 = os.path.join(dataset_path_inference, 'pcd0', fn)
filepath_pcd1 = os.path.join(dataset_path_inference, 'pcd1', fn)
filepath_pcd2 = os.path.join(dataset_path_inference, 'pcd2', fn)
filepath_pcd3 = os.path.join(dataset_path_inference, 'pcd3', fn)
filepath_pcdc = os.path.join(dataset_path_inference, 'pcdc', fn)

with h5py.File(filepath_partial, 'r') as f:
    pcd_partial = np.array(f['data']).astype(np.float64)
with h5py.File(filepath_gt, 'r') as f:
    pcd_gt = np.array(f['data']).astype(np.float64)
with h5py.File(filepath_pcd0, 'r') as f:
    pcd0 = np.array(f['data']).astype(np.float64)
with h5py.File(filepath_pcd1, 'r') as f:
    pcd1 = np.array(f['data']).astype(np.float64)
with h5py.File(filepath_pcd2, 'r') as f:
    pcd2 = np.array(f['data']).astype(np.float64)
with h5py.File(filepath_pcd3, 'r') as f:
    pcd3 = np.array(f['data']).astype(np.float64) 
with h5py.File(filepath_pcdc, 'r') as f:
    pcdc = np.array(f['data']).astype(np.float64)

### Init plot function
app = Plot3DApp()

fig1 = app.create_figure(figure_name='Fig 1', viewports_dim=(1,7), width=1280, height=720, sync_camera=False, plot_boundary=True, show_axes=True, show_subtitles=True, background_color=(1,1,1,1), snapshot_path='./')

fig1a = fig1.set_subplot(0,0,'Partial Point Cloud')
pcvis.draw_pc_xyz(fig1a, pcd_partial, (0,0,1))
fig1b = fig1.set_subplot(0,1,'pcd0')
pcvis.draw_pc_xyz(fig1b, pcd0, (0,0,1))
fig1c = fig1.set_subplot(0,2,'pcd1')
pcvis.draw_pc_xyz(fig1c, pcd1, (0,0,1))
fig1d = fig1.set_subplot(0,3,'pcd2')
pcvis.draw_pc_xyz(fig1d, pcd2, (0,0,1))
fig1e = fig1.set_subplot(0,4,'pcd3')
pcvis.draw_pc_xyz(fig1e, pcd3, (0,0,1))
fig1f = fig1.set_subplot(0,5,'pcdc')
pcvis.draw_pc_xyz(fig1f, pcdc, (0,0,1))
fig1g = fig1.set_subplot(0,6,'Complete Point Cloud')
pcvis.draw_pc_xyz(fig1g, pcd_gt, (1,0,0))


### Start GUI
fig1.ready() # must call ready!

app.plot()
app.close()
