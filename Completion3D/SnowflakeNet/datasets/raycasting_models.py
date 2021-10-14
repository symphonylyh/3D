'''
Use raycasting to generate partial and complete point clouds. 

My first attempt was to use the Unity-generated stockpile and associate each instance with its prototype model. However, the orientation is inconsistent between the partial and complete models, therefore the training results is not satifactory.

Then the idea would be to simulate LiDAR in Unity for each individual rock and obtain the particle and complete clouds. It may take longer since it's in C#. 

Finally, I found Open3D has raycasting fucntionality in the latest release (> 0.13.0). We need to pip install the latest version. This is good!
H:\envs\snowflake\Scripts\pip.exe install --user --pre https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-0.13.0+299f29e-cp37-cp37m-win_amd64.whl. Currently this functionality is under open3d.t.geometry, in the near future we may need to update it.
'''

import os,sys,random, colorsys
import open3d as o3d
import pymeshlab as ml
import plyfile
import numpy as np
import h5py
import matplotlib.pyplot as plt

def random_colors(N, bright=True, seed=0):
    brightness = 1.0 if bright else 0.7
    hsv = [(i/float(N), 1, np.random.uniform(0.7,1.0)) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors
    
class RayCastingModel:
    def __init__(self, filepath, result_path, snapshot_path):
        '''
        Module for 3D model --> partial & complete cloud analysis. The class instance applies on a per-model basis.

        :param str filepath path of the .ply model
        :param str result_path path of the folder to save results
        '''
        self.model_path = filepath # full path
        self.model_name = os.path.splitext(os.path.basename(filepath))[0] # RR3_1
        self.result_path = result_path
        self.snapshot_path = snapshot_path
    
    def read_model_init_scene(self):
        '''
        Read .ply/.obj model and re-center at the origin, and initialize the raycasting scene.
        '''
        self.mesh = o3d.io.read_triangle_mesh(self.model_path)

        # re-center at the origin
        self.mesh.translate(translation=np.array([0,0,0]), relative=False) # relative False means move the centroid to the translation vector rather than offset

        # estimate model radius
        bbox = self.mesh.get_oriented_bounding_box().extent
        self.radius_model = (3/4 * bbox[0] * bbox[1] * bbox[2] /np.pi)**(1/3)

        # convert to o3d.t.geometry mesh
        self.tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        
        # create scene
        self.scene = o3d.t.geometry.RaycastingScene()
        self.mesh_id = self.scene.add_triangles(self.tmesh)

    def _equidistribution_on_sphere_surface(self, N, r=1):
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

    def init_full_lidars(self, N, r=1):
        '''
        Init full LiDAR locations that cover 360-degree distribution on a sphere surface of given radius.

        :param int N number of LiDARs
        :param float r radius
        '''
        self.lidar_radius = r
        self.lidar_full = self._equidistribution_on_sphere_surface(N, r=r)
        self.N_lidar_full = self.lidar_full.shape[0]

    def init_partial_lidars(self, num_partials):
        '''
        Init partial LiDAR locations that are a subset of the full lidars

        :param int [N_start, ..., N_end] number of partial LiDARs, i.e., the first N (40%-80%) of the full LiDAR list.
        '''
        self.lidar_partial = num_partials
        self.N_lidar_partial = len(num_partials)

    def raycasting_full(self, N_ring, ring_spacing=0.1, arc_spacing=0.1):
        '''
        Do raycasting of the full LiDAR set. This simulates each lidar projecting a cluster of rays described as rings. We can also use the built-in create_rays_pinhole() but it's less under our control.

        :param int N_ring number of rings
        :param float ring_spacing spacing of the ring along the radius direction
        :param float arc_spacing spacing of the ray points along the arc direction
        '''
        lidar_locs = self.lidar_full[:,:3]
        lookat = np.array([0,0,0])
        self.raypoints = []
        self.rayhits = []
        for i in range(self.N_lidar_full):
            lidar_loc = lidar_locs[i]
            
            # plane equation
            # given the normal vector (a,b,c) of a plane and a known point (x0,y0,z0) on the plane, the equation of the plane that passes throught the point & perpendicular to the normal vector is given by: a(x-x0) + b(y-y0) + c(z-z0) = 0 --> based on the principle that normal (a,b,c) is orthogonal to any arbitrary vectors on the plane (x-x0,y-y0,z-z0)
            # in our case, normal is lidar_loc - lookat, known point is lookat
            n = lidar_loc - lookat
            p = lookat
            
            ### circle equation on the plane 
            # ref: https://www.quora.com/A-problem-in-3D-geometry-what-is-the-equation-of-the-circle-see-details
            # suppose (x,y,z) is a point on the circle with radius r
            # (x,y,z) = (x0,y0,z0) + rcos(t)*u + rsin(t)*v, where u,v are orthonormal basis of the plane
            # to find orthonormal basis, we can (a) find two arbitrary linearly indenpendent vectors and do Gram-Schmidt, or (b) find one arbitrary vector and cross product with normal.
            # see details in note, here we used (b)
            
            # find orthonormal basis
            # find non-zero field in the normal (a,b,c):
            nonzero_idx = 0
            for idx in range(3):
                if n[idx] > 1e-6: # caveat: float precision
                    nonzero_idx = idx
                    break
            assert n[nonzero_idx] != 0, "Wrong normal vector"
            # find u
            u = np.array([1.0,1.0,1.0])
            dp = 0
            for idx in range(3):
                if idx != nonzero_idx:
                    dp += n[idx] * (u[idx]-p[idx]) # dot product of the other two components
            u[nonzero_idx] = -dp / n[nonzero_idx] + p[nonzero_idx]
            # find v by cross product
            v = np.cross(n, u)
            # normalize to make orthonormal basis
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)

            # circle equation
            rtheta = np.empty((0,2),float)
            for ring_i in range(1,N_ring+1):
                r = ring_i * ring_spacing
                # based on fixed arc length, determine the theta spacing on the ring
                # arc length S = r * d_theta, d_theta in radian
                d_theta = arc_spacing / r
                N_theta = int(2*np.pi / d_theta)
                d_theta = 2*np.pi / N_theta
                temp = np.column_stack( (np.ones(N_theta) * r, np.arange(N_theta) * d_theta) )
                rtheta = np.concatenate((rtheta, temp), axis=0)
            
            self.raypoints.append( lookat + rtheta[:,0].reshape(-1,1) * (np.outer(np.cos(rtheta[:,1]), u) + np.outer(np.sin(rtheta[:,1]),v)) ) # * is element-wise multiply, should use outer or matrix form multiply, otherwise we need to do reshape a lot
            print(f'LiDAR {i} has {self.raypoints[i].shape[0]} raypoints')

            raydirections = self.raypoints[i] - lidar_loc
            raydirections = raydirections / np.sqrt(np.sum(raydirections**2, axis=1))[:,None] # normalize to unit vector
            rays = np.concatenate((np.tile(lidar_loc, (len(self.raypoints[i]), 1)), raydirections), axis=1)
            rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            rayhits = self.scene.cast_rays(rays)
            # problem with Open3D current functinality is it doesn't return the hit point, instead it returns the intersection distance in the unit of the direction vector, so we need to extract the hit point by ourselves
            rayhits = rayhits['t_hit'].numpy()
            hitflags = np.invert(np.isinf(rayhits))
            rayhits = rayhits[hitflags]
            raydirections = raydirections[hitflags]
            rayhits = lidar_loc + rayhits[:,None] * raydirections
            self.rayhits.append(rayhits)
            print(f'LiDAR {i} has {self.rayhits[i].shape[0]} rayhits')

    def permute_orientation(self):
        pass 

    ### Visualization functinalities
    def vis_lidar_locations(self, display=False, save_snapshots=True):
        '''
        Visualize different lidar locations.

        :param bool display flag to display Open3D windows
            - window display the lidar positions (full & partial) with model at center
        :param bool save_snapshots flag to save intermediate results
            - figure showing lidar position on sphere surface , 'RRX_N_.png'
        '''
        vis_cam = o3d.visualization.Visualizer()
        vis_cam.create_window('LiDARs', width=1280, height=1280, visible=display)

        # draw a reference sphere surface
        # due to Open3D capability, we need to first generate a triangle mesh --> line set --> visualize
        sphere_surface = o3d.geometry.TriangleMesh.create_sphere(radius=self.lidar_radius)
        sphere_surface.translate(translation=np.array([0,0,0]), relative=False)
        sphere_surface1 = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_surface)
        sphere_surface1.paint_uniform_color(np.array([0.6,0.6,0.6])) # gray
        # sphere_surface2 = o3d.geometry.PointCloud(sphere_surface.vertices) # only show points
        vis_cam.add_geometry(sphere_surface1)
    
        # look-at camera model
        # ref: https://ksimek.github.io/2012/08/22/extrinsic/
        view_control = vis_cam.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(np.array([-1,1,0.5])) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction

        # draw model
        vis_cam.add_geometry(self.mesh.scale(scale=1.0,center=np.array([0,0,0])), reset_bounding_box=False) # prevent viewpoint from changing automatically

        # draw LiDAR locations and save snapshots
        j = 0
        for i in self.lidar_partial + [self.N_lidar_full]:
            # add lidar
            while j < i:
                cam_j = o3d.geometry.TriangleMesh.create_sphere(radius=self.lidar_radius/50)
                arrow_j = o3d.geometry.TriangleMesh.create_arrow()
                arrow_j.translate(translation=self.lidar_full[j,:3], relative=False)
                arrow_j.rotate( o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0,0,0]-self.lidar_full[j,:3])) )
                cam_j.translate(translation=self.lidar_full[j,:3], relative=False)
                cam_j.paint_uniform_color(np.array([0.9,0.5,0])) # orange
                vis_cam.add_geometry(cam_j)   
                vis_cam.add_geometry(arrow_j)
                j += 1
            # save
            snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
            if save_snapshots:
                plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidars_{j}.png'), snapshot)
            
        if display:
            vis_cam.run()
        vis_cam.destroy_window()

    def vis_lidar_view(self, display=True, save_snapshots=True):
        pass 

    def vis_lidar_rays(self, display=True, save_snapshots=True):
        '''
        Visualize different lidar rays.

        :param bool display flag to display Open3D windows
            - window display the camera positions with model at center
        :param bool save_snapshots flag to save intermediate results
            - figure showing camera position on sphere surface (with coordinate frame at center), 'camera_position.png'
        '''
        vis_cam = o3d.visualization.Visualizer()
        vis_cam.create_window('LiDAR Rays', width=1280, height=1280, visible=display)
        colors = random_colors(self.N_lidar_full)

        for i in range(self.N_lidar_full):
            lidar_loc = self.lidar_full[i,:3]
            cam_i = o3d.geometry.TriangleMesh.create_sphere(radius=self.lidar_radius/50)
            cam_i.translate(translation=self.lidar_full[i,:3], relative=False)
            if i < self.N_lidar_partial:
                cam_i.paint_uniform_color(np.array([0.9,0.5,0])) # orange
            else:
                cam_i.paint_uniform_color(np.array([0,0.1,0.6])) # blue
            vis_cam.add_geometry(cam_i)   
            
            # show raycasting endpoints
            # raypoints = self.raypoints[i]
            # pc1 = o3d.geometry.PointCloud()
            # pc1.points = o3d.utility.Vector3dVector(raypoints) # from numpy to o3d format
            # pc1.paint_uniform_color(colors[i])
            # vis_cam.add_geometry(pc1)
            # o3d.io.write_point_cloud(os.path.join(self.snapshot_path, 'test1.ply'), pc1)

            # line_pts = np.concatenate((lidar_loc.reshape(1,-1), raypoints), axis=0)
            # line_indices = [[0, pt+1] for pt in range(len(raypoints))] # from lidar pos to all its points
            # rays = o3d.geometry.LineSet()
            # rays.points = o3d.utility.Vector3dVector(line_pts)
            # rays.lines = o3d.utility.Vector2iVector(line_indices)
            # rays.paint_uniform_color(colors[i])
            # vis_cam.add_geometry(rays)

            # show ray hits
            rayhits = self.rayhits[i]
            print(rayhits.shape, rayhits)
            pc2 = o3d.geometry.PointCloud()
            pc2.points = o3d.utility.Vector3dVector(rayhits) # from numpy to o3d format
            pc2.paint_uniform_color(colors[i])
            vis_cam.add_geometry(pc2)
            # o3d.io.write_point_cloud(os.path.join(self.snapshot_path, 'test2.ply'), pc2)
            # break

        # look-at camera model
        # ref: https://ksimek.github.io/2012/08/22/extrinsic/
        view_control = vis_cam.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(np.array([-1,1,0.5])) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction

        # snapshot: lidar positions with model
        # vis_cam.add_geometry(self.mesh.scale(scale=1.0,center=np.array([0,0,0])), reset_bounding_box=False) # prevent viewpoint from changing automatically
        snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
        if save_snapshots:
            plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidar_position_{self.N_lidar_full}_{self.N_lidar_partial}.png'), snapshot)

        if display:
            vis_cam.run()
        vis_cam.destroy_window()

if __name__ == '__main__':
    root_path = 'H:/RockScan'
    rock_category = 'RR3'
    start_folder = 1
    end_folder = 46
    result_path = "H:/git_symphonylyh/3D/Completion3D/SnowflakeNet/datasets/rocks3d/train"
    snapshot_path = "H:/git_symphonylyh/3D/Completion3D/SnowflakeNet/datasets/rocks3d/vis"
    # hyper-parameters
    N_orientations = 1 # number of different particle orientation views
    N_lidar_full = 16 # number of full LiDARs
    N_lidar_partial = list( range(int(N_lidar_full*0.4), int(N_lidar_full*0.8)+1) ) # use the first N (40%-80%) of the full LiDARs
    ring_spacing = 0.01
    arc_spacing = 0.01
    num_points_per_partial = 2048
    num_points_per_gt = 16384 # same as ShapeNet PCN

    for folderID in range(start_folder, end_folder+1):
        fn = rock_category+'_'+str(folderID)+'.ply'
        print(f'Processing {fn}')
        model_path = os.path.join(root_path, rock_category, str(folderID), 'models')
        filepath = os.path.join(model_path, fn)

        r = RayCastingModel(filepath, result_path, snapshot_path)
        r.read_model_init_scene()

        ### Initialize LiDARs
        # LiDARs settings remains constant for one model among different orienations, but could change from model to model (because the models are not of the same size)
        N_ring = int(1.5 * r.radius_model / ring_spacing) # ring covers the extent of the particle
        r.init_full_lidars(N=N_lidar_full, r=5*r.radius_model) # lidars are placed 5 times distance of the model radius
        r.init_partial_lidars(num_partials=N_lidar_partial)
        print(f'Using {r.N_lidar_full} full LiDARs and {r.lidar_partial} partial LiDARs')

        ### Visualization
        # this is done once per model, i.e., only save snapshots for the first orientation, otherwise it's too many repeated snapshots
        # (1) visualize lidar locations for each lidar set (partial lidars in orange)
        r.vis_lidar_locations(display=False, save_snapshots=True)
        #
        # r.vis_lidar_rays(display=False, save_snapshots=False)
        
        # repeat
        # for each model: 
        #   init LiDAR sets
        #   for each orientation:
        #       raycast one full LiDAR set
        #       for different # of selected partial LiDARs:
        #           raycast several partial LiDARs sets
        # for o in range(N_orientations):
        #     # simulate one full LiDAR set
            
        #     # simulate several partial LiDARs sets
        #     # permute the particle orientation & move to next
        #     r.permute_orientation()
           

        #     r.raycasting_full(N_ring=N_ring, ring_spacing=ring_spacing, arc_spacing=arc_spacing)
            
        break

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

        break

    
    