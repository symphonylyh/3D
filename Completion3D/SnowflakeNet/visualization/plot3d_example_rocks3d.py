'''
Example for visualizing Rock3D shape completion data.
Usage:
    - python plot3d_example_rocks3d.py
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
import open3d as o3d
import pymeshlab as ml

from plot3d import Plot3DApp, Plot3DFigure, PointCloudVis as pcvis

###  Read point cloud file
dataset_path_partial = '../datasets/rocks3d/test/partial/001/'
dataset_path_gt = '../datasets/rocks3d/test/gt/001/'
file_id = 120
dataset_path_inference = '../datasets/rocks3d/test/benchmark/001/'

files = [os.path.basename(f) for f in os.listdir(dataset_path_partial)]
fn = files[file_id]
filestem = os.path.splitext(fn)[0].split('_', 1)[-1]
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

def compute_point_cloud_volume_and_area(pcd, method='PS'):
    '''
    Compute the volume of a complete point cloud by first reconstructing the mesh. 
    
    Notes:
        - The input point cloud must be a complete cloud that can form a closed surface!
        - The input point cloud must be re-centered at the origin!

    :param ndarray (N,3) pcd raw coordinates of the point cloud.
    :param str method method for surface reconstruction. 'BP' (Ball Pivoting) or 'PS' (Poisson's Surface Reconstruction). For complete cloud in meshlab, I found PS is more stable because it gurantees closed mesh for volume measurement.
    Ref: https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba

    Note: the mesh reconstruction in Open3D is slow and error-prone... (such as missing point normals, non-watertight mesh, etc.). So finally I decided to use pymeshlab
    '''
    ### Final version (pymeshlab)
    m = ml.Mesh(pcd)
    ms = ml.MeshSet()
    ms.add_mesh(m, 'pcd')
    ms.compute_normals_for_point_sets()
    if method == 'BP':
        ms.surface_reconstruction_ball_pivoting()
    elif method == 'PS':
        ms.surface_reconstruction_screened_poisson()
    ms.close_holes()
    ms.invert_faces_orientation(forceflip=True) # somehow the normals are always pointing inwards
    # ms.poisson_disk_sampling(samplenum=num_points_per_gt, exactnumflag=True) # in case we want to simplify the mesh

    # calculate geometric features: bbox length, volume and area
    measures = ms.compute_geometric_measures()
    bbox = measures['bbox']
    bbox_dim = (bbox.dim_x(), bbox.dim_y(), bbox.dim_z())
    volume = measures['mesh_volume']
    area = measures['surface_area']
    # print("[PRINT] %d vertices, %d faces, area %.2f cm^2, volume %.2f cm^3" % (ms.current_mesh().vertex_number(), ms.current_mesh().face_number(), measures['surface_area']*1e4, measures['mesh_volume']*1e6) )
    save_model_name = 'test.ply'
    ms.save_current_mesh(file_name=save_model_name, binary=False, save_vertex_normal=True, save_vertex_color=True)

    return bbox_dim, volume, area

    ### Old version (Open3D, not stable)
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(pcd)
    # pc.estimate_normals() # normal is necessary for reconstruction
    
    # ### from point cloud to mesh 
    # if method == 'BP':
    #     # estimate the ball radii
    #     distances = pc.compute_nearest_neighbor_distance()
    #     avg_distance = np.mean(distances) # average distance from a point to its nearest neighbor
    #     radius = 3 * avg_distance
    #     radii = o3d.utility.DoubleVector([radius, 2 * radius])
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, radii)
    # elif method == 'PS':
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    #     # in case the mesh is not a close mesh, we can remove the bulb shape by cropping the original bbox. This is usually not necessary in our rock case
    #     # bbox = pc.get_axis_aligned_bounding_box()
    #     # mesh = mesh.crop(bbox)

    # ## in case we need to simplify the mesh to a target number of faces
    # # mesh = mesh.simplify_quadric_decimation(1000)
    # ## in case we need to fix artifacts in the mesh
    # # mesh.remove_degenerate_triangles()
    # # mesh.remove_duplicated_triangles()
    # # mesh.remove_duplicated_vertices()
    # # mesh.remove_non_manifold_edges()
    # ## currently there is no hole-filling implementation in Open3D. If that's needed, we should use pymeshlab

    # ### compute volume
    # volume = mesh.get_volume()
    # return volume

volume, area = compute_point_cloud_volume_and_area(pcd3, method='PS')

# ### Init plot function
# app = Plot3DApp()

# fig1 = app.create_figure(figure_name='Fig 1', viewports_dim=(1,7), width=1280, height=720, sync_camera=True, plot_boundary=True, show_axes=False, show_subtitles=True, background_color=(1,1,1,2), snapshot_path='./', snapshot_prefix=filestem)
# # background color (1,1,1,0) is pure black, (1,1,1,2) is pure white. weird

# point_size=5
# fig1a = fig1.set_subplot(0,0,'Partial Point Cloud \n(N=2048)')
# pcvis.draw_pc_xyz(fig1a, pcd_partial, (0,0,1), point_size=point_size)
# fig1f = fig1.set_subplot(0,1,'Coarse Seeds \n(N=256)')
# pcvis.draw_pc_xyz(fig1f, pcdc, (0,0,1), point_size=point_size)
# fig1b = fig1.set_subplot(0,2,'Sparse Cloud P0 \n(N=512)')
# pcvis.draw_pc_xyz(fig1b, pcd0, (0,0,1), point_size=point_size)
# fig1c = fig1.set_subplot(0,3,'Rearranged Cloud P1 \n(N=512)')
# pcvis.draw_pc_xyz(fig1c, pcd1, (0,0,1), point_size=point_size)
# fig1d = fig1.set_subplot(0,4,'Upsampled Cloud P2 \n(N=2048')
# pcvis.draw_pc_xyz(fig1d, pcd2, (0,0,1), point_size=point_size)
# fig1e = fig1.set_subplot(0,5,'Upsampled Cloud P3 \n(N=16,384)')
# pcvis.draw_pc_xyz(fig1e, pcd3, (0,0,1), point_size=point_size)
# fig1g = fig1.set_subplot(0,6,'Ground-Truth Point Cloud \n(N=16,384)')
# pcvis.draw_pc_xyz(fig1g, pcd_gt, (1,0,0), point_size=point_size)


# ### Start GUI
# fig1.ready() # must call ready!

# app.plot()
# app.close()

# from vis_splitting import *
# path12 = splitting_paths(pcd1, pcd2, inds=np.arange(10), colors_points=(0,0.1,0.6), colors_paths=(1,0,0), points_radius=0.001, paths_radius=0.0005)
# o3d.visualization.draw_geometries([path12], window_name="Path12", point_show_normal=True)

# path123 = splitting_paths_triple(pcd1, pcd2, pcd3, inds=np.arange(10), colors_points=(0,0.1,0.6), colors_path1=(1,0,0), colors_path2=(0.9,0.5,0), points_radius=0.001, paths_radius1=0.0005, paths_radius2=0.00025)
# o3d.visualization.draw_geometries([path123], window_name="Path123", point_show_normal=True)

# path123_range = splittings_by_range(pcd1, pcd2, pcd3, range_x=(0,0.4), range_y=(0,0.4), range_z=(0,0.4), colors_points=(0,0.1,0.6), colors_path1=(1,0,0), colors_path2=(0.9,0.5,0), points_radius=0.001, paths_radius1=0.0005, paths_radius2=0.00025)
# o3d.visualization.draw_geometries([path123_range], window_name="Path123_Range", point_show_normal=True)
