'''
After running the inference on test set, we have generated results in a 'benchmark' folder, we then visualize the shape completion results & compare with the ground-truth shapes. 
'''

import os, sys
import h5py
import numpy as np
import open3d as o3d
import pymeshlab as ml
import matplotlib.pyplot as plt 
import pandas as pd 
import openpyxl

from visualization.plot3d import Plot3DApp, Plot3DFigure, PointCloudVis as pcvis
from visualization.vis_splitting import *

class CompareShape:
    def __init__(self, input_path, output_path, gt_path, snapshot_path):
        '''
        Module for comparing the partial shape (input), completed shape (output), and ground-truth shape (gt). Each input/output/gt_path contains models with the same filename.
        ''' 
        self.input_path = input_path
        self.output_path = output_path
        self.gt_path = gt_path
        self.snapshot_path = snapshot_path
    
        # get all file names
        self.files = [os.path.basename(f) for f in os.listdir(self.input_path)] # filenames
        self.filestems = [os.path.splitext(fn)[0].split('_', 1)[-1] for fn in self.files] # filestems by stripping the leading uuid and file extension

    def load_shapes(self, file_id):
        ''' 
        Load all shapes of a model.

        :param int file_id
        ''' 
        fn = self.files[file_id]
        filepath_partial = os.path.join(self.input_path, fn)

        filepath_pcdc = os.path.join(self.output_path, 'pcdc', fn)
        filepath_pcd0 = os.path.join(self.output_path, 'pcd0', fn)
        filepath_pcd1 = os.path.join(self.output_path, 'pcd1', fn)
        filepath_pcd2 = os.path.join(self.output_path, 'pcd2', fn)
        filepath_pcd3 = os.path.join(self.output_path, 'pcd3', fn)

        filepath_gt = os.path.join(self.gt_path, fn)

        with h5py.File(filepath_partial, 'r') as f:
            pcd_partial = np.array(f['data']).astype(np.float64)
       
        with h5py.File(filepath_pcdc, 'r') as f:
            pcdc = np.array(f['data']).astype(np.float64)
        with h5py.File(filepath_pcd0, 'r') as f:
            pcd0 = np.array(f['data']).astype(np.float64)
        with h5py.File(filepath_pcd1, 'r') as f:
            pcd1 = np.array(f['data']).astype(np.float64)
        with h5py.File(filepath_pcd2, 'r') as f:
            pcd2 = np.array(f['data']).astype(np.float64)
        with h5py.File(filepath_pcd3, 'r') as f:
            pcd3 = np.array(f['data']).astype(np.float64) 
    
        with h5py.File(filepath_gt, 'r') as f:
            pcd_gt = np.array(f['data']).astype(np.float64)
        
        return pcd_partial, pcdc, pcd0, pcd1, pcd2, pcd3, pcd_gt

    def vis_shape_completion(self, file_id, pcd_partial, pcdc, pcd0, pcd1, pcd2, pcd3, pcd_gt, point_size=5):
        ''' 
        Visualize the progress of shape completion.

        :param int file_id
        :param ndarray point cloud at each stage.
        :param float point_size

        Note: this is interactive method. It display a window showing the progress of shape completion with sync views. And to save snapshots, the user should adjust the views to a desirable angle, and press 'S', then figure of clouds at each stage will be saved as 'RRX_N_subtitle_name.png'.
        '''
        filestem = self.filestems[file_id]

        app = Plot3DApp()

        fig1 = app.create_figure(figure_name='Shape Completion', viewports_dim=(1,7), width=720*7, height=720, sync_camera=True, plot_boundary=True, show_axes=False, show_subtitles=True, background_color=(1,1,1,2), snapshot_path=self.snapshot_path, snapshot_prefix=filestem)
        # background color (1,1,1,0) is pure black, (1,1,1,2) is pure white. weird

        # all clouds are blue except the gt cloud in red
        cloud_color = (0,0,1)
        gt_cloud_color = (1,0,0)
        fig1a = fig1.set_subplot(0,0,'Partial Point Cloud \n(N=2048)')
        pcvis.draw_pc_xyz(fig1a, pcd_partial, cloud_color, point_size=point_size)
        fig1f = fig1.set_subplot(0,1,'Coarse Seeds \n(N=256)')
        pcvis.draw_pc_xyz(fig1f, pcdc, cloud_color, point_size=point_size)
        fig1b = fig1.set_subplot(0,2,'Sparse Cloud P0 \n(N=512)')
        pcvis.draw_pc_xyz(fig1b, pcd0, cloud_color, point_size=point_size)
        fig1c = fig1.set_subplot(0,3,'Rearranged Cloud P1 \n(N=512)')
        pcvis.draw_pc_xyz(fig1c, pcd1, cloud_color, point_size=point_size)
        fig1d = fig1.set_subplot(0,4,'Upsampled Cloud P2 \n(N=2048')
        pcvis.draw_pc_xyz(fig1d, pcd2, cloud_color, point_size=point_size)
        fig1e = fig1.set_subplot(0,5,'Upsampled Cloud P3 \n(N=16,384)')
        pcvis.draw_pc_xyz(fig1e, pcd3, cloud_color, point_size=point_size)
        fig1g = fig1.set_subplot(0,6,'Ground-Truth Point Cloud \n(N=16,384)')
        pcvis.draw_pc_xyz(fig1g, pcd_gt, gt_cloud_color, point_size=point_size)


        ### Start GUI
        fig1.ready() # must call ready!

        app.plot()
        app.close()

    def vis_splitting_paths(self, file_id, pcd1, pcd2, pcd3):
        ''' 
        Visualize the splitting paths in SnowflakeNet.

        To save snapshots, the user should adjust the views to a desirable angle, and press 'S', then figure of splitting path will be saved as 'RRX_N_splitting_paths.png'.
        '''
        # path12 = splitting_paths(pcd1, pcd2, inds=np.arange(10), colors_points=(0,0.1,0.6), colors_paths=(1,0,0), points_radius=0.001, paths_radius=0.0005)
        # o3d.visualization.draw_geometries([path12], window_name="Path12", point_show_normal=True)

        # path123 = splitting_paths_triple(pcd1, pcd2, pcd3, inds=np.arange(10), colors_points=(0,0.1,0.6), colors_path1=(1,0,0), colors_path2=(0.9,0.5,0), points_radius=0.001, paths_radius1=0.0005, paths_radius2=0.00025)
        # o3d.visualization.draw_geometries([path123], window_name="Path123", point_show_normal=True)

        path123_range = splittings_by_range(pcd1, pcd2, pcd3, range_x=(0,0.4), range_y=(0,0.4), range_z=(0,0.4), colors_points=(0,0.1,0.6), colors_path1=(1,0,0), colors_path2=(0.9,0.5,0), points_radius=0.001, paths_radius1=0.0005, paths_radius2=0.00025)
        # visualization with key callback
        # ref: https://github.com/isl-org/Open3D/blob/f775cae33c517b53433ef70fdf1cb4ae3919c4d5/examples/python/visualization/customized_visualization.py#L77
        snapshot_name = os.path.join(self.snapshot_path, self.filestems[file_id]+'_splitting_paths.png')
        def capture_image(vis):
            snapshot = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            plt.imsave(snapshot_name, snapshot)
            return False
        key_to_callback = {}
        key_to_callback[ord("S")] = capture_image
        o3d.visualization.draw_geometries_with_key_callbacks([path123_range], key_to_callback=key_to_callback, window_name="Path123_Range")
        # o3d.visualization.draw_geometries([path123_range], window_name="Path123_Range", point_show_normal=True)

    def pcd_to_mesh(self, pcd, method='PS'):
        '''
        Reconstruct point cloud to mesh. 
    
        Notes:
            - The input point cloud must be a complete cloud that can form a closed surface!

        :param ndarray (N,3) pcd raw coordinates of the point cloud.
        :param str method method for surface reconstruction. 'BP' (Ball Pivoting) or 'PS' (Poisson's Surface Reconstruction). For complete cloud in meshlab, I found PS is more stable because it gurantees closed mesh for volume measurement.
        Ref: https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba

        Note: the mesh reconstruction in Open3D is slow and error-prone... (such as missing point normals, non-watertight mesh, etc.). So finally I decided to use pymeshlab
        '''
        # re-center the pcd at origin
        pcd -= pcd.mean(axis=0)

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

        # Note: some mesh's normals are randomly inward/outward, we want it always be outward, so we check the volume (if volume is negative, it means the normals are inward)
        if ms.compute_geometric_measures()['mesh_volume'] < 0:
            ms.invert_faces_orientation(forceflip=True) 
        # ms.poisson_disk_sampling(samplenum=num_points_per_gt, exactnumflag=True) # in case we want to simplify the mesh

        return ms
        
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

    def geometric_measures(self, ms):
        '''
        Measure the size and shape properties.

        Note: assume coords is meter! Some properties we report in cm
        :param Pymeshlab MeshSet ms 
        '''
        # calculate geometric features: bbox length, volume and area
        measures = ms.compute_geometric_measures()
        bbox = measures['bbox']
        bbox_dim = (bbox.dim_x(), bbox.dim_y(), bbox.dim_z())
        bbox_dim = np.sort(bbox_dim)
        volume = measures['mesh_volume']
        area = measures['surface_area']

        ESD = (3/4 / np.pi * volume)**(1/3) * 2
        FER3D = bbox_dim[2] / bbox_dim[0]
        Sphericity3D = (36*np.pi * volume**2)**(1/3) / area
        
        return {'ESD (cm)': ESD*1e2, 'a (cm)': bbox_dim[0]*1e2, 'b (cm)': bbox_dim[1]*1e2, 'c (cm)': bbox_dim[2]*1e2, 'Volume (cm^3)': volume*1e6, 'Area (cm^2)': area*1e4, 'FER3D': FER3D,  'Sphericity3D': Sphericity3D}

    def save_mesh(self, ms, name):
        ms.save_current_mesh(file_name=os.path.join(self.snapshot_path, name), binary=False, save_vertex_normal=True, save_vertex_color=True)

    def compare_one(self, file_id, vis=True, save_mesh=True, debug_print=True):
        ''' 
        Compare the completion and ground-truth of one model.
        '''
        fn = self.files[file_id]
        print(f"Dataset has {len(self.files)} samples, comparing sample {file_id}: {fn}")

        # load point cloud
        pcd_partial, pcdc, pcd0, pcd1, pcd2, pcd3, pcd_gt = self.load_shapes(file_id)

        # visualization
        if vis:
            # shape completion progress with sync views (only run once for illustration)
            self.vis_shape_completion(file_id, pcd_partial, pcdc, pcd0, pcd1, pcd2, pcd3, pcd_gt,  point_size=5)

            # splitting paths
            self.vis_splitting_paths(file_id, pcd1, pcd2, pcd3)

        # convert to mesh
        mesh_complete = self.pcd_to_mesh(pcd3)
        mesh_gt = self.pcd_to_mesh(pcd_gt)

        # compute geometric measures
        geometric_complete = self.geometric_measures(mesh_complete)
        geometric_gt = self.geometric_measures(mesh_gt)

        # print
        df_complete = pd.DataFrame.from_dict(geometric_complete, orient='index', columns=['Completion'])
        df_gt = pd.DataFrame.from_dict(geometric_gt, orient='index', columns=['GT'])
        df_all = pd.concat([df_complete, df_gt], axis=1)
        if debug_print:
            print(df_all)

        # save mesh
        if save_mesh:
            mesh_name = self.filestems[file_id] 
            self.save_mesh(mesh_complete, mesh_name + '_complete.ply')
            self.save_mesh(mesh_gt, mesh_name + '_gt.ply')
        
        return df_complete, df_gt

    def compare_all(self, save_spreadsheet_name, vis=False, save_mesh=False, debug_print=False):
        ''' 
        Compare the completion and ground-truth of all models in the test dataset, and save the statistics to spreadsheet.
        '''
        df_complete, df_gt = [], []
        for fid in range(len(self.files)):
            df1, df2 = self.compare_one(file_id=fid, vis=vis, save_mesh=save_mesh, debug_print=debug_print)
            df_complete.append(df1)
            df_gt.append(df2)
        
        df_complete_all = pd.concat(df_complete, axis=1)   
        df_gt_all = pd.concat(df_gt, axis=1)  
        df_complete_all.set_axis(self.filestems, axis=1, inplace=True)
        df_gt_all.set_axis(self.filestems, axis=1, inplace=True)

        # write results to spreadsheet1
        spreadsheet = os.path.join(self.snapshot_path, save_spreadsheet_name)
        with pd.ExcelWriter(spreadsheet, mode='w') as writer:
            df_complete_all.to_excel(writer, sheet_name='Completion', float_format='%.3f')
            df_gt_all.to_excel(writer, sheet_name='Ground-Truth', float_format='%.3f')

    def plot_all(self, save_spreadsheet_name, show=False):
        ''' 
        Analyze the spreadsheet from previous compare_all() and plot.
        '''
        spreadsheet = os.path.join(self.snapshot_path, save_spreadsheet_name)
        pred = pd.read_excel(spreadsheet, sheet_name='Completion')
        gt = pd.read_excel(spreadsheet, sheet_name='Ground-Truth')

        row_index = [0,4,5,6,7]
        row_name = [r'ESD (cm)', r'$Volume\ (cm^3)$', r'$Area\ (cm^2)$', r'$FER_{3D}$', r'$Sphericity_{3D}$']
        plot_name = ['ESD.png', 'Volume.png', 'Area.png', 'FER3D.png', 'Sphericity3D.png']

        data_pred, data_gt, MAPE = [], [], []
        for row in row_index:
            pred_np = pred.iloc[row,1:].dropna().to_numpy()            
            gt_np = gt.iloc[row,1:].dropna().to_numpy()
            error = np.abs(pred_np - gt_np) / gt_np * 100
            error = np.sum(error) / gt_np.shape[0]

            data_pred.append(pred_np)
            data_gt.append(gt_np)
            MAPE.append(error)
        
        for i, name in enumerate(plot_name):
            fig = plt.figure()
            markersize = 3
            pass_point = np.min(data_gt[i] * 0.95)
            plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')
            plt.plot(data_gt[i], data_pred[i], 'o', markerfacecolor='none', markersize=markersize)
            plt.gca().set_aspect('equal')

            plt.text(0.7,0.2,f'MAPE={MAPE[i]:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

            plt.xlabel(row_name[i]+r', Ground-Truth')
            plt.ylabel(row_name[i]+r', Prediction')
            plt.grid()
            fig.savefig(os.path.join(self.snapshot_path, name), bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close(fig)

if __name__ == '__main__':
    input_path = './datasets/rocks3d/test/partial/001/'
    output_path = './datasets/rocks3d/test/benchmark/001/'
    gt_path = './datasets/rocks3d/test/gt/001/'
    snapshot_path = './datasets/rocks3d/vis/completion/'

    c = CompareShape(input_path, output_path, gt_path, snapshot_path)

    # for debug checking one file at a time
    # c.compare_one(file_id=183, vis=False, save_mesh=True, debug_print=True)
    
    save_spreadsheet_name = 'shape_completion_comparison.xlsx'

    # run all files
    # c.compare_all(save_spreadsheet_name=save_spreadsheet_name)

    c.plot_all(save_spreadsheet_name=save_spreadsheet_name)