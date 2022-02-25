''' 
High-level 3D stockpile instance segmentation module.
'''

import os, sys, random
import h5py
import numpy as np
import open3d as o3d
import pymeshlab as ml
import plyfile
import matplotlib.pyplot as plt 
import pandas as pd 
import openpyxl

from visualization.plot3d import Plot3DApp, Plot3DFigure, PointCloudVis as pcvis
import torch
from models.model import SnowflakeNet as Model


class Completion3D:
    def __init__(self, model_path):
        self.model_path = model_path
    
    def load_model(self):
        self.model = Model(dim_feat=512, up_factors=[4, 8])
      
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).cuda()

        # Load the pretrained model from a checkpoint
        print(f'Recovering from {self.model_path}')
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])

        # Switch models to evaluation mode
        self.model.eval()

    def inference(self, partial):
        with torch.no_grad():
            partial = torch.from_numpy(partial).float()
            pcds = self.model(partial, return_P0=True)
            pcdc, pcd0, pcd1, pcd2, pcd3 = pcds
            pcdc = pcdc.squeeze().cpu().numpy()
            pcd0 = pcd0.squeeze().cpu().numpy()
            pcd1 = pcd1.squeeze().cpu().numpy()
            pcd2 = pcd2.squeeze().cpu().numpy()
            pcd3 = pcd3.squeeze().cpu().numpy()
        return pcdc, pcd0, pcd1, pcd2, pcd3

class Segmentation3D:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.filestem = os.path.splitext(os.path.basename(input_path))[0]
        self.output_path = output_path

    def raw_cloud_block_divide(self):
        pass 

    def raw_cloud_downsample(self, num_points):
        ''' 
        Downsample the raw point cloud to a sparse cloud that be better processed by the segmentation network.
        
        :param int num_points target number of points for downsampling. Not exact, but it doesn't matter.
        '''
        pass 
    
    def segmentation(self):
        pass 

    def load_results(self):
        ''' 
        Load segmentation result PLY file.

        :param str filename 
        '''
        filename = self.input_path
        f = plyfile.PlyData().read(filename)
        self.results = np.array([list(x) for x in f.elements[0]]).astype(np.float32)
        
        self.coords = self.results[:, :3]
        self.colors = self.results[:, 3:6].astype(np.int32)
        self.sem_labels = self.results[:, 6].astype(np.int32)
        self.ins_labels = self.results[:, 7].astype(np.int32)
        self.ins_scores = self.results[:, 8]
        self.offsets = self.results[:, 9:12]
        
        self.results_shift = np.array(self.results)
        self.results_shift[:,:3] = self.results_shift[:,:3] + self.offsets

    def vis_results(self, display=False, save_snapshots=True, sp=None, sp_threshold=0):
        # apply shape percentage thresholding
        if sp_threshold > 0:
            print(f'Display instances with Shape Percentage > {int(sp_threshold*100)}')
            ins_labels = self.results[:,7]
            for ins in range(len(sp)):
                if sp[ins] < sp_threshold:
                    print(f'ins {ins} less than sp threshold')
                    self.results[np.argwhere(ins_labels == ins)[:,0], 7] = -1

        pc_xyzrgb = self.results[:,:6][np.newaxis,:,:]
        pc_xyzrgbsemins = self.results[:,:8][np.newaxis,:,:]
        pc_xyzrgbsemins_shift = self.results_shift[:,:8][np.newaxis,:,:]

        app = Plot3DApp()

        fig1 = app.create_figure(figure_name='Fig 1', viewports_dim=(1,3), width=1920, height=720, sync_camera=True, plot_boundary=True, show_axes=False, show_subtitles=True, background_color=(1,1,1,2), snapshot_path=self.output_path, snapshot_prefix=self.filestem)
        fig2 = app.create_figure(figure_name='Fig 2', viewports_dim=(1,1), width=1280, height=720, sync_camera=True, plot_boundary=False, show_axes=False, show_subtitles=True, background_color=(1,1,1,2), snapshot_path=self.output_path, snapshot_prefix=self.filestem)
        
        fig1a = fig1.set_subplot(0,0,'Raw Point Cloud')
        pcvis.draw_pc_raw(fig1a, pc_xyzrgb)
        # fig1b = fig1.set_subplot(0,1,'Point Cloud by Semantic')
        # pcvis.draw_pc_by_semins(fig1b, pc_xyzrgbsemins, sem_dict=None, color_code='semantic', show_legend=True)
        fig1b = fig1.set_subplot(0,1,'Point Cloud by Instance (shifted coordinates)')
        pcvis.draw_pc_by_semins(fig1b, pc_xyzrgbsemins_shift, sem_dict=None, color_code='instance', show_bbox=False)
        fig1c = fig1.set_subplot(0,2,'Point Cloud by Instance')
        pcvis.draw_pc_by_semins(fig1c, pc_xyzrgbsemins, sem_dict=None, color_code='instance', show_bbox=True, line_width=2)

        fig2a = fig2.set_subplot(0,0,'Point Cloud by Instance Label')
        pcvis.draw_pc_by_semins(fig2a, pc_xyzrgbsemins, sem_dict=None, color_code='instance', show_bbox=True, bbox_axis_align=True, bbox_color='black', show_instance_label=True, line_width=2)

        fig1.ready()
        fig2.ready()
        if save_snapshots:
            fig1.save_snapshots()
        if display:
            app.plot()
        app.close()

    def vis_shape_completion(self, ins_id, pcd_partial, pcdc, pcd0, pcd1, pcd2, pcd3, point_size=5, display=False, save_snapshots=True):
        ''' 
        Visualize the progress of shape completion.

        :param int ins_id instance ID in the stockpile.
        :param ndarray point cloud at each stage.
        :param float point_size

        Note: this is interactive method. It display a window showing the progress of shape completion with sync views. And to save snapshots, the user should adjust the views to a desirable angle, and press 'S', then figure of clouds at each stage will be saved as 'RRX_N_subtitle_name.png'.
        '''
        app = Plot3DApp()

        fig1 = app.create_figure(figure_name='Shape Completion', viewports_dim=(1,6), width=360*7, height=360, sync_camera=True, plot_boundary=True, show_axes=False, show_subtitles=True, background_color=(1,1,1,2), snapshot_path=self.output_path, snapshot_prefix=self.filestem+'_'+str(ins_id).zfill(3))
        # background color (1,1,1,0) is pure black, (1,1,1,2) is pure white. weird

        # all clouds are blue except the gt cloud in red
        cloud_color = (0,0,1)
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

        ### Start GUI
        fig1.ready() # must call ready!
        if save_snapshots:
            fig1.save_snapshots()
        if display:
            app.plot()
        app.close()

    def extract_instances_and_complete(self, completion_model, display=False, save_snapshots=True, save_mesh=True, save_spreadsheet=True):
        partial_folder = os.path.join(self.output_path, 'partial')
        complete_folder = os.path.join(self.output_path, 'complete')
        if not os.path.exists(partial_folder):
            os.makedirs(partial_folder)
        if not os.path.exists(complete_folder):
            os.makedirs(complete_folder)

        min_ins_points = 100
        num_points_per_ins = 2048
        ins_keys = np.unique(self.ins_labels)

        # [IMPORTANT!] in our output from the segmentation step, we gave non-instance points an instance label of -1, we need to omit these
        ins_keys = ins_keys[ins_keys > -1] 

        num_valid_instances = 0
        df_complete_all = []
        for ins_id in ins_keys:
            print(f'Processing instance {ins_id+1}/{len(ins_keys)} in the stockpile')

            ins_points = self.coords[self.ins_labels == ins_id,:]

            # only consider instance having enough points (can add shape percentage check here as well! e.g. only consider instance with shape > 40% percentage)
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
            
            # [IMPORTANT step] center the resampled points at origin! 
            ins_points_resampled = ins_points_resampled - ins_points_resampled.mean(axis=0) # shift to start at origin

            # run completion inference 
            pcdc, pcd0, pcd1, pcd2, pcd3 = completion_model.inference(ins_points_resampled[np.newaxis,:,:])  # add new axis to be like a batch_size=1

            # visualization
            if display or save_snapshots:
                self.vis_shape_completion(ins_id, ins_points_resampled, pcdc, pcd0, pcd1, pcd2, pcd3, point_size=5, display=display, save_snapshots=save_snapshots)
            
            # convert to mesh
            mesh_complete = self.pcd_to_mesh(pcd3, downsample=2048)

            # save partial cloud, complete cloud, and complete mesh
            if save_mesh:
                save_fn = self.filestem + '_' + str(ins_id).zfill(3)

                partial_pcd_path = os.path.join(partial_folder, save_fn+'_partial.ply')
                complete_pcd_path = os.path.join(complete_folder, save_fn+'_complete.ply')
                complete_mesh_path = os.path.join(complete_folder, save_fn+'_complete_mesh.ply')

                self.save_cloud(ins_points_resampled, partial_pcd_path)
                self.save_cloud(pcd3, complete_pcd_path)
                self.save_mesh(mesh_complete, complete_mesh_path)

            # compute geometric measures
            geometric_complete = self.geometric_measures(mesh_complete)

            df = pd.DataFrame.from_dict(geometric_complete, orient='index', columns=[str(ins_id)])
            df_complete_all.append(df)
            
            # break

        # write results to spreadsheet
        if save_spreadsheet:
            df_complete_all = pd.concat(df_complete_all, axis=1) 
            save_spreadsheet_name = self.filestem + '.xlsx'
            spreadsheet = os.path.join(self.output_path, save_spreadsheet_name)
            with pd.ExcelWriter(spreadsheet, mode='w') as writer:
                df_complete_all.to_excel(writer, sheet_name='Completion', float_format='%.3f')

        print(f'{self.ins_labels.shape[0]} points, {num_valid_instances}/{len(ins_keys)} valid instances')

    def pcd_to_mesh(self, pcd, downsample=2048, method='PS'):
        '''
        Reconstruct point cloud to mesh. 
    
        First implementation in benchmark_rocks3d.py
        '''
        # re-center the pcd at origin
        pcd -= pcd.mean(axis=0)

        ### Final version (pymeshlab)
        m = ml.Mesh(pcd)
        ms = ml.MeshSet()
        ms.add_mesh(m, 'pcd')
        ms.point_cloud_simplification(samplenum=downsample, exactnumflag=True)
        ms.compute_normals_for_point_sets()
        if method == 'BP':
            ms.surface_reconstruction_ball_pivoting()
        elif method == 'PS':
            ms.surface_reconstruction_screened_poisson()
        elif method == 'CH':
            ms.convex_hull()
        ms.close_holes()

        # Note: some mesh's normals are randomly inward/outward, we want it always be outward, so we check the volume (if volume is negative, it means the normals are inward)
        measures = ms.compute_geometric_measures()
        if 'mesh_volume' in measures.keys() and measures['mesh_volume'] < 0:
            ms.invert_faces_orientation(forceflip=True) 
        # ms.poisson_disk_sampling(samplenum=num_points_per_gt, exactnumflag=True) # in case we want to simplify the mesh
        return ms
        
    def geometric_measures(self, ms):
        '''
        Measure the size and shape properties.

        Note: assume coords is meter! Some properties we report in cm
        :param Pymeshlab MeshSet ms 
        '''
        # calculate geometric features: bbox length, volume and area
        measures = ms.compute_geometric_measures()
        # bbox = measures['bbox']
        # bbox_dim = (bbox.dim_x(), bbox.dim_y(), bbox.dim_z())
        # bbox_dim = np.sort(bbox_dim)
        # FER3D = bbox_dim[2] / bbox_dim[0]
        # [IMPORTANT] for bbox, we should NOT use meshlab's axis-aligned bbox, instead we should use open3d's oriented bbox
        pcd = ms.mesh(1).vertex_matrix()
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pcd))
        bbox_dim = bbox.extent 
        bbox_dim = np.sort(bbox_dim)
        FER3D = bbox_dim[2] / bbox_dim[0]

        valid = True
        if 'mesh_volume' in measures.keys():
            volume = measures['mesh_volume']
            area = measures['surface_area']
            ESD = (3/4 / np.pi * volume)**(1/3) * 2
            Sphericity3D = (36*np.pi * volume**2)**(1/3) / area
        else:
            print('Problematic mesh, cannot calculate volume!')
            valid = False
            volume = 0
            area = 0
            ESD = 0
            Sphericity3D = 0
        
        return {'ESD (cm)': ESD*1e2, 'a (cm)': bbox_dim[0]*1e2, 'b (cm)': bbox_dim[1]*1e2, 'c (cm)': bbox_dim[2]*1e2, 'Volume (cm^3)': volume*1e6, 'Area (cm^2)': area*1e4, 'FER3D': FER3D,  'Sphericity3D': Sphericity3D}

    def save_cloud(self, pcd, path):
        pc0 = o3d.geometry.PointCloud()
        pc0.points = o3d.utility.Vector3dVector(pcd)
        o3d.io.write_point_cloud(path, pc0)
    
    def save_mesh(self, ms, path):
        ms.save_current_mesh(file_name=path, binary=True, save_vertex_normal=True, save_vertex_color=True)

if __name__ == '__main__':
    results_path = 'H:/AggregateStockpile/segmentation_results'
    model_path = './exp/checkpoints/2021-10-15T03-30-37/ckpt-best.pth'

    inspection_mode = True # for inspection, it just displays the interactive window for me to identify the instance correspondence to ground-truth, without saving any results
    sp_threshold_mode = True # show shape percentage thresholding effect
    
    if not inspection_mode:
    # load trained shape completion model 
        c = Completion3D(model_path=model_path)
        c.load_model()
    
    filelist = [ply for ply in os.listdir(results_path) if ply.endswith('.ply')]
    
    start_id = 22
    end_id = 22
    for fid in range(start_id, end_id + 1):
        f = filelist[fid]
        result_folder = os.path.join(results_path, os.path.splitext(f)[0])
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        print(f'Analyzing file {f}')
        s = Segmentation3D(input_path=os.path.join(results_path, f), output_path=result_folder)
        
        s.load_results()

        if sp_threshold_mode:
            stem = os.path.splitext(f)[0]
            spreadsheet_path = os.path.join(results_path, stem, stem+'.xlsx')
            sp = pd.read_excel(spreadsheet_path, sheet_name='Completion').iloc[8,1:].dropna().to_numpy() # shape percentage
            s.vis_results(display=True, save_snapshots=False, sp=sp, sp_threshold=0.80)
            break
        
        if not inspection_mode:
            s.vis_results(display=False, save_snapshots=True)
        else:
            s.vis_results(display=True, save_snapshots=False)

        if not inspection_mode:
            s.extract_instances_and_complete(c, display=False, save_snapshots=False, save_mesh=True, save_spreadsheet=False)        

