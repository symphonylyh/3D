
import os, sys, random
import h5py
import numpy as np
import open3d as o3d
import pymeshlab as ml
import plyfile
import matplotlib.pyplot as plt 
import pandas as pd 
import openpyxl
import copy 

from visualization.plot3d import Plot3DApp, Plot3DFigure, PointCloudVis as pcvis

class ShapePercentage:
    def __init__(self, sphere_mesh_file='directional_sphere_1000.ply'):
        self.dirs = self.equidistribution_on_sphere_surface(N=1000, r=1)[:,:3] # unit vectors
        self.sphere_mesh = o3d.io.read_triangle_mesh(sphere_mesh_file)        

        ### directional sphere mesh was generated in advance
        # we need to visualize the shape percentage concept on a directional sphere
        # we want to have a unit sphere of N=1000 uniformly-distributed vertices, where each vertex represents a direction to check if it hits a surface element
        # now it's hard to have the mesh of such sphere, because open3D surface reconstruction is trash!
        # finally we decided to export the point cloud to meshlab and obtain the mesh by ball pivoting, then read the mesh in open3D
        
        # save sphere mesh (only do once)
        # pcd_sphere = o3d.geometry.PointCloud()
        # pcd_sphere.points = o3d.utility.Vector3dVector(self.dirs)
        # pcd_sphere.estimate_normals()
        # pcd_sphere.orient_normals_towards_camera_location(centroid)
        # pcd_sphere.normals = o3d.utility.Vector3dVector(-np.asarray(pcd_sphere.normals))
        # o3d.io.write_point_cloud('directional_sphere_1000.ply', pcd_sphere)
        
        # open3D surface reconstruction is trash!
        # - ball pivoting always has holes
        # radii = [0.16, 0.24, 0.32, 0.4, 0.48]
        # self.sphere_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_sphere,o3d.utility.DoubleVector(radii)) # good thing about ball pivoting is it doesn't add more vertices during reconstruction
        # - Poisson reconstruction adds new vertices
        # self.sphere_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_sphere, depth=8, width=0, scale=1.1, linear_fit=False)[0]

    def shape_percentage(self, pcd, method='point', display=True, save_snapshots=True, snapshot_name=None):
        '''
        Calculate shape percentage of a partial shape.
        pcd Open3D PointCloud
        method str: 'mesh' for Poisson reconstructed surface; 'Point' for point projection
        '''
        # recenter point cloud at origin (although we already did this in segementation3D)
        pcd.paint_uniform_color((0,0.1,0.6)) # blue
        centroid = pcd.get_center()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) - centroid.mean(axis=0)) 
        percentage = 0

        if method == 'mesh': # this method is not very accurate since we crop by bbox
            # computer point normals (since we need to reconstruct a surface mesh of the partial point cloud, normals are required)
            pcd.estimate_normals()
            pcd.orient_normals_towards_camera_location(centroid) # press 'n' to show normals, '-'/'+' to adjust normal length
            pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
            
            poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
            bbox = pcd.get_oriented_bounding_box()
            p_mesh_crop = poisson_mesh.crop(bbox)
            p_mesh_crop.paint_uniform_color([0.5,0.5,0.5])
            # o3d.visualization.draw_geometries([p_mesh_crop], width=640, height=640, mesh_show_wireframe=True, mesh_show_back_face=True)

            rays = np.concatenate((np.tile(centroid, (len(self.dirs), 1)), self.dirs), axis=1) # (origin, direction)
            rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

            # create scene
            scene = o3d.t.geometry.RaycastingScene()
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(p_mesh_crop)
            mesh_id = scene.add_triangles(tmesh)

            rayhits = scene.cast_rays(rays)
            rayhits = rayhits['t_hit'].numpy()
            hitflags = np.invert(np.isinf(rayhits))
            rayhits = rayhits[hitflags]
            num_hits = np.count_nonzero(hitflags)
            
        elif method == 'point':
            # compute average point-to-point distance in the point cloud
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            hit_threshold = avg_dist

            # ray hit test
            vec_a = np.asarray(pcd.points) # vector OA (origin to point cloud), Nx3
            vec_b = self.dirs.T # vector OB (origin to unit sphere direction), 3xM
            # a.b = |a||b|cost --> orthogonal projection |a|cost = a.b/|b| where |b|=1
            acost = vec_a @ vec_b
            # what's ab (or acost)?
            # we use matrix multiply to calculate dot product. ab[:,0] is a.b for all points vec_a and direction 0, ab[:,1] is for direction 1, and so on
            anorm = np.linalg.norm(vec_a,axis=1).reshape(-1,1)
            asint = np.sqrt(anorm**2 - acost**2) # this is point distance to the direction vector

            # criteria:
            # 1 - acost > 0, acost is signed projection, where positive means the angle between vec_a and vec_b is less than 90 degrees, i.e. in the same half-space
            # 2 - asint < threshold, asint is point-to-line distance, less than certain threshold means this direction hits the point cloud (shape existence)
            hitflags = np.logical_and(acost > 0, asint < hit_threshold) # NxM

            # collapse hit flags. Why? if we look the NxM column-wise, each column means given a direction, whether there is points that hit by the ray. So, as long as there is one True in the column, the ray has a hit!
            hitflags = np.any(hitflags, axis=0)
            num_hits = np.count_nonzero(hitflags)
        
        # visualize directional sphere
        # estimate model radius
        bbox = pcd.get_oriented_bounding_box().extent
        pcd_radius = (3/4 * bbox[0] * bbox[1] * bbox[2] /np.pi)**(1/3)
        sphere_mesh = copy.deepcopy(self.sphere_mesh)
        sphere_mesh.scale(1/3 * pcd_radius, (0,0,0)) # scale the directional sphere
        sphere_mesh.paint_uniform_color(np.array([0.9,0.5,0])) # orange for missing part
        sphere_vertex_colors = np.asarray(sphere_mesh.vertex_colors)
        sphere_vertex_colors[hitflags,:] = np.array([0,0.1,0.6]) # blue for existing part
        sphere_mesh.vertex_colors = o3d.utility.Vector3dVector(sphere_vertex_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window('Shape Percentage', width=1280, height=1280, visible=display)
        vis.add_geometry(sphere_mesh)
        vis.add_geometry(pcd)
        view_control = vis.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(-self.dirs[hitflags,:][0,:]) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction

        render = vis.get_render_option()
        render.light_on = False
        render.mesh_show_back_face = True
        render.mesh_show_wireframe = True 
        vis.update_renderer()
        
        snapshot = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        if save_snapshots:
            plt.imsave(snapshot_name, snapshot)
        
        if display:
            vis.run()
        vis.destroy_window()

        percentage = num_hits / len(self.dirs)
        print(f'{num_hits}/{len(self.dirs)} hits, percentage {percentage*100}%')

        return percentage

    def equidistribution_on_sphere_surface(self, N, r=1):
        '''
        Generate uniformly distributed points on a unit sphere surface.
        Ref: https://scicomp.stackexchange.com/a/30005
        Method 1: random placement in (z,fi) interval
        Method 2: regular placement, distance between two points in two orthogonal directions is locally same.

        :param int N number of points
        :param float r radius

        :return ndarray point coordinates (x,y,z,theta,phi), [N,5]. (theta, phi) is used later on to calculate cross product

        Notes: first implementation in the projection analysis projection.py
        '''

        # spherical coordinates (r,theta,phi)
        # right-handed coordinate system (x,y,z)
        #     z
        #     |
        #     |__ __ y
        #    /
        #   /
        #  x  
        # - r: radius of the sphere
        # - theta: polar angle (vertical), [0,pi], 0 at +z axis, rotate clockwise
        # - phi: azimuthal angle (horizontal), [0,2pi], 0 at +x axis, rotate counter-clockwise
        # x = r * sin(theta) * cos(phi)
        # y = r * sin(theta) * sin(phi)
        # z = r * cos(theta)

        # Method 1: discretizing theta and z dimension - not evenly spaced
        # Method 2: 
        # (a) latitude circles at constant theta intervals
        # (b) on each latitude circle, place points with constant phi intervals
        # (c) such that the distance in theta and phi are almost the same 
        area = 4.0*np.pi*r**2 / N # average area per point
        dist = np.sqrt(area)
        N_theta = int(np.round(np.pi*r/dist)) # pi*r is half circumference, divide by target distance
        d_theta = np.pi*r / N_theta
        d_phi = area / d_theta
        N_generated = 0
        points = []
        for i in range (N_theta): # generate different latitude circles
            theta = np.pi*(i+0.5)/N_theta
            N_phi = int(np.round(2*np.pi*r*np.sin(theta)/d_phi))
            for j in range (N_phi): # generate points along each latitude circle
                phi = 2*np.pi*j/N_phi
                x = r*np.sin(theta)*np.cos(phi)
                y = r*np.sin(theta)*np.sin(phi)
                z = r*np.cos(theta)
                points.append((x,y,z,theta,phi))
                N_generated += 1
        points = np.array(points)

        return points

if __name__ == '__main__':
    sphere_mesh_file='directional_sphere_1000.ply'
    s = ShapePercentage(sphere_mesh_file)

    root_path = 'H:/AggregateStockpile/segmentation_results'
    stockpilelist = [ply for ply in os.listdir(root_path) if ply.endswith('.ply')]
    start_sid = 0
    end_sid = 2
    for sid in range(start_sid, end_sid + 1):
        sname = stockpilelist[sid]
        filestem = os.path.splitext(sname)[0]
        results_path = os.path.join(root_path, filestem, 'partial')

        filelist = [ply for ply in os.listdir(results_path) if ply.endswith('.ply')]
        start_id = 0
        end_id = len(filelist) - 1
        df_all = []
        for fid in range(start_id, end_id + 1):
            f = filelist[fid]
            print(f'Analyzing file {f}')

            # read partial point cloud
            pcd = o3d.io.read_point_cloud(os.path.join(results_path, f))

            percentage = s.shape_percentage(pcd, method='point', display=False, save_snapshots=True, snapshot_name=os.path.join(results_path, os.path.splitext(f)[0]+'.png'))

            df = pd.DataFrame.from_dict({'Shape Percentage': percentage}, orient='index', columns=[str(fid)])
            df_all.append(df)

        # save in a separate spreadsheet
        df_all = pd.concat(df_all, axis=1) 
        save_spreadsheet_name = 'shape_percentage.xlsx'
        spreadsheet = os.path.join(root_path, filestem, save_spreadsheet_name)
        with pd.ExcelWriter(spreadsheet, mode='w') as writer:
            df_all.to_excel(writer, sheet_name='Percentage', float_format='%.3f')









