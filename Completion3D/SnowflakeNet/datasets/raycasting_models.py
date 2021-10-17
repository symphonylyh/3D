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
import uuid 

def random_colors(N, bright=True, seed=0):
    brightness = 1.0 if bright else 0.7
    hsv = [(i/float(N), 1, np.random.uniform(0.7,1.0)) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    
    Ref: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

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
            

        raypoints_per_lidar = self.raypoints[0].shape[0]
        avg_rayhits = sum([len(h) for h in self.rayhits]) / len(self.rayhits)
        # print(f'Each of {self.N_lidar_full} LiDARs casts {raypoints_per_lidar} raypoints and has {int(avg_rayhits)} rayhits on average')

    def save_clouds(self, permutation, partial_points=2048, complete_points=16384):
        '''
        Save partial and complete clouds in h5 format, re-sampling to a target number of points.

        :param int permutation X-th permutation of the particle orientation
        :param int partial_points fixed number of partial points.
        :param int complete_points fixed number of complete points.
        '''
        def point_resampling_exact(cloud, target_num):
            '''
            Minor fix the number of points to exactly what we want by pad or crop.
            Note: the current number should already be close to the target number.
            '''
            if len(cloud) == target_num:
                return cloud
            if len(cloud) > target_num:
                # crop
                cloud =cloud[:target_num,:]
            else:
                # pad
                padding = random.sample(list(range(len(cloud))), target_num - len(cloud))
                selected_idx = list(range(len(cloud))) + padding
                cloud = cloud[selected_idx, :]
            return cloud

        j = 0
        cloud = np.empty((0,3), float)
        complete_cloud = None
        save_fns = []
        for i in self.lidar_partial + [self.N_lidar_full]:
            # accumulate lidar data
            while j < i:
                cloud = np.concatenate((cloud, self.rayhits[j]), axis=0)
                j += 1
            
            # partial clouds
            if i < self.N_lidar_full:
                # re-sample partial points
                assert len(cloud) >= partial_points, "Raw partial points fewer than expected!"
                # use random downsampling to simulate sensor inaccuracy
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(cloud)
                pc = pc.random_down_sample(sampling_ratio=partial_points/len(cloud))
                pc = np.asarray(pc.points)
                pc = point_resampling_exact(pc, target_num=partial_points)

                uid = uuid.uuid4().hex
                save_fn = f'{uid}_{self.model_name}_l{str(i)}_p{str(permutation).zfill(3)}.h5'
                save_fns.append(save_fn)
                partial_path = os.path.join(self.result_path, 'partial', '001', save_fn)
                with h5py.File(partial_path, 'w') as f:
                    f.create_dataset('data', data=pc)
                
                # debug print ply
                # pc0 = o3d.geometry.PointCloud()
                # pc0.points = o3d.utility.Vector3dVector(pc)
                # o3d.io.write_point_cloud(os.path.join(self.snapshot_path, f'{self.model_name}_l{str(i)}_p{str(permutation).zfill(3)}_partial.ply'), pc0)
            # complete cloud
            else:
                # re-sample complete points
                assert len(cloud) >= complete_points, "Raw complete points fewer than expected!"
                # use random downsampling to simulate sensor inaccuracy
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(cloud)
                pc = pc.random_down_sample(sampling_ratio=complete_points/len(cloud))
                pc = np.asarray(pc.points)
                pc = point_resampling_exact(pc, target_num=complete_points)
                complete_cloud = pc
                # save later since we need duplicates
                break
                
        for i, lidar_i in enumerate(self.lidar_partial):
            save_fn = save_fns[i]
            gt_path = os.path.join(self.result_path, 'gt', '001', save_fn)
            with h5py.File(gt_path, 'w') as f:
                f.create_dataset('data', data=complete_cloud)

            # debug print ply
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(complete_cloud)
            # o3d.io.write_point_cloud(os.path.join(self.snapshot_path, f'{self.model_name}_l{str(i)}_p{str(permutation).zfill(3)}_gt.ply'), pc)

    def compute_orientations(self, N):
        '''
        Compute the uniformly distributed orientation vectors.

        :param int N number of target orientations.
        '''
        orientation_vec = self._equidistribution_on_sphere_surface(N, r=1)[:,:3] # if r=1, they should be unit vector already, and if the lookat is origin, it's by itself the directional vector
        return orientation_vec

    def permute_orientation(self, old_orientation, new_orientation):
        '''
        Permuate the model orientation. Since Open3D can only rotate the model by relative rotation, we need to explicit calculate the relative rotation between the permutation.

        :param (3,) old_orientation
        :param (3,) new_orientation
        '''
        R = rotation_matrix_from_vectors(old_orientation, new_orientation)
        self.mesh = self.mesh.rotate(R, center=[0,0,0])
        # update the raycasting scene
        self.tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.mesh_id = self.scene.add_triangles(self.tmesh)

    ### Visualization functinalities
    def vis_lidar_locations(self, display=False, save_snapshots=True):
        '''
        Visualize different lidar locations.

        :param bool display flag to display Open3D windows
            - window display the lidar positions (full) with model at center
        :param bool save_snapshots flag to save intermediate results
            - figure showing lidar position on sphere surface , a sequence of 'RRX_N_lidars_M.png' for M being the number of lidars in partial and full.
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
    
        # draw model
        vis_cam.add_geometry(self.mesh.scale(scale=1.0,center=np.array([0,0,0])), reset_bounding_box=False) # prevent viewpoint from changing automatically

        # look-at camera model
        # ref: https://ksimek.github.io/2012/08/22/extrinsic/
        view_control = vis_cam.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(np.array([-1,1,0.5])) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction

        # draw LiDAR locations and save snapshots
        j = 0
        for i in self.lidar_partial + [self.N_lidar_full]:
            # add lidar
            while j < i:
                cam_j = o3d.geometry.TriangleMesh.create_sphere(radius=self.lidar_radius/50)
                cam_j.compute_vertex_normals()
                cam_j.translate(translation=self.lidar_full[j,:3], relative=False)
                cam_j.paint_uniform_color(np.array([0.9,0.5,0])) # orange
                # create arrow
                # Caveat: Open3D arrow inits at pointing direction (0,0,1), and the rotate() is relative instead of absolute. Also remember that the center of arrow is not at the endpoint but the mid of the cylinder, so the translate destination is not directly the position.
                arrow_len = 0.03 # default cylinder height is 5.0, so the scale factor is
                scale_factor = arrow_len / 5.0
                arrow_j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.25, cone_radius=1.0)
                arrow_j.compute_vertex_normals() # this allows Phong shading!
                direction = np.array([0,0,0])-self.lidar_full[j,:3]
                direction /= np.linalg.norm(direction)
                R = rotation_matrix_from_vectors(np.array([0,0,1]), direction) # compute relative rotation from the default direction (0,0,1) to the lookat vector direction
                arrow_j.rotate(R)
                arrow_j.scale(scale=scale_factor, center=np.array([0,0,0]))
                arrow_j.translate(translation=self.lidar_full[j,:3] + direction * arrow_len/2, relative=False)
                arrow_j.paint_uniform_color(np.array([0.9,0.5,0])) # orange
                vis_cam.add_geometry(cam_j, reset_bounding_box=False)   
                vis_cam.add_geometry(arrow_j, reset_bounding_box=False)
                j += 1

            # save
            snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
            if save_snapshots:
                plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidars_{j}.png'), snapshot)
            
        if display:
            vis_cam.run()
        vis_cam.destroy_window()

    def vis_lidar_rays(self, display=False, save_snapshots=True):
        '''
        Visualize lidar rays.

        :param bool display flag to display Open3D windows
            - window1 display lidar rays
        :param bool save_snapshots flag to save intermediate results
            - figure showing lidar raycasting disk, 'lidar_raydisk.png'. This is only saved once for illustration
            - figure showing lidar rayhits on the rock surface, 'RRX_N_lidari_rayhits.png'
        '''
        # window1: show only one LiDAR rays (w/ endpoints and hitpoints)
        i = 1 # lidar id for visualization
        vis_cam = o3d.visualization.Visualizer()
        vis_cam.create_window('LiDAR Ray Disk', width=1280, height=1280, visible=display)
        colors = random_colors(self.N_lidar_full)

        ### Snapshot 1: lidar raycasting endpoints (without model). Since this is same for all cases, I just save it once for lidar=1 and commented it
        # draw lidar
        lidar_loc = self.lidar_full[i,:3]
        cam_i = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius_model/20)
        cam_i.compute_vertex_normals()
        cam_i.translate(translation=self.lidar_full[i,:3], relative=False)
        cam_i.paint_uniform_color(np.array([0.9,0.5,0])) # orange
        vis_cam.add_geometry(cam_i)   
        
        # draw raycasting endpoints
        raypoints = self.raypoints[i]
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(raypoints) # from numpy to o3d format
        pc1.paint_uniform_color(np.array([0,0.1,0.6])) # blue
        vis_cam.add_geometry(pc1)
        
        # draw rays (from lidar to endpoints)
        line_pts = np.concatenate((lidar_loc.reshape(1,-1), raypoints), axis=0)
        line_indices = [[0, pt+1] for pt in range(len(raypoints))] # from lidar pos to all its points
        rays1 = o3d.geometry.LineSet()
        rays1.points = o3d.utility.Vector3dVector(line_pts)
        rays1.lines = o3d.utility.Vector2iVector(line_indices)
        rays1.paint_uniform_color(np.array([0.5,0.5,0.5])) # gray
        vis_cam.add_geometry(rays1)

        # look-at camera model
        # ref: https://ksimek.github.io/2012/08/22/extrinsic/
        view_control = vis_cam.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(np.array([-1,1,0.5])) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction
        view_control.set_zoom(1.2) # > 1 means zoom out

        # # snapshot
        # snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
        # if save_snapshots:
        #     plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidar_raydisk.png'), snapshot)
        
        ### Snapshot 2: lidar raycasting hitpoints (with model)
        vis_cam.remove_geometry(pc1, reset_bounding_box=False)
        vis_cam.remove_geometry(rays1, reset_bounding_box=False)

        # draw model
        vis_cam.add_geometry(self.mesh.scale(scale=1.0,center=np.array([0,0,0])), reset_bounding_box=False) # prevent viewpoint from changing automatically

        # draw ray hits
        rayhits = self.rayhits[i]
        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(rayhits) # from numpy to o3d format
        pc2.paint_uniform_color(np.array([0,0.1,0.6])) # blue
        vis_cam.add_geometry(pc2, reset_bounding_box=False)

        # draw rays
        line_pts = np.concatenate((lidar_loc.reshape(1,-1), rayhits), axis=0)
        line_indices = [[0, pt+1] for pt in range(len(rayhits))] # from lidar pos to all its points
        rays2 = o3d.geometry.LineSet()
        rays2.points = o3d.utility.Vector3dVector(line_pts)
        rays2.lines = o3d.utility.Vector2iVector(line_indices)
        rays2.paint_uniform_color(np.array([0.5,0.5,0.5])) # gray
        vis_cam.add_geometry(rays2, reset_bounding_box=False)

        snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
        if save_snapshots:
            plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidar{i}_rayhits.png'), snapshot)

        if display:
            vis_cam.run()
        vis_cam.destroy_window()

    def vis_lidar_multi_views(self, display=False, save_snapshots=True):
        '''
        Visualize multi-view lidar rays.

        :param bool display flag to display Open3D windows
            - window1 display multi-view lidar rays
        :param bool save_snapshots flag to save intermediate results
            - figure showing two-view lidar with model, 'RRX_N_lidar_i_j_multiview_w_model.png'
            - figure showing two-view lidar without model, 'RRX_N_lidar_i_j_multiview_wo_model.png'
            - figure showing two-view lidar without ray but with viewing direction, 'RRX_N_lidar_i_j_multiview_wo_ray.png'
        '''
        lidar_ids = [1,3]  # lidar id for visualization
        colors = [[0.9,0.5,0], [0,0.1,0.6]] # orange, blue
        vis_cam = o3d.visualization.Visualizer()
        vis_cam.create_window('Multi-view LiDAR', width=1280, height=1280, visible=display)
        # vis_cam.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        
        # draw model
        model = self.mesh
        model.scale(scale=1.0,center=np.array([0,0,0]))
        vis_cam.add_geometry(model, reset_bounding_box=False) # prevent viewpoint from changing automatically
        
        rays = []
        for i, lidar_i in enumerate(lidar_ids):
            # draw lidar
            lidar_loc = self.lidar_full[lidar_i,:3]
            cam_i = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius_model/20)
            cam_i.compute_vertex_normals()
            cam_i.translate(translation=lidar_loc, relative=False)
            cam_i.paint_uniform_color(np.array(colors[i]))
            vis_cam.add_geometry(cam_i)  

            # draw ray hits
            rayhits = self.rayhits[lidar_i]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(rayhits) # from numpy to o3d format
            pc.paint_uniform_color(np.array(colors[i])) 
            vis_cam.add_geometry(pc, reset_bounding_box=True)

            # draw rays
            line_pts = np.concatenate((lidar_loc.reshape(1,-1), rayhits), axis=0)
            line_indices = [[0, pt+1] for pt in range(len(rayhits))] # from lidar pos to all its points
            rays0 = o3d.geometry.LineSet()
            rays0.points = o3d.utility.Vector3dVector(line_pts)
            rays0.lines = o3d.utility.Vector2iVector(line_indices)
            # rays.paint_uniform_color(np.array([0.5,0.5,0.5])) # gray
            rays0.paint_uniform_color(np.array(colors[i])) # gray
            rays.append(rays0)
            vis_cam.add_geometry(rays[i], reset_bounding_box=True)

        # look-at camera model
        # ref: https://ksimek.github.io/2012/08/22/extrinsic/
        view_control = vis_cam.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(np.array([-0.5,3,0.5])) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction
        view_control.set_zoom(1) # > 1 means zoom out

        # snapshot1: multiview with model
        snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
        if save_snapshots:
            plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidar_{lidar_ids[0]}_{lidar_ids[1]}_multiview_w_model.png'), snapshot)

        # snapshot2: multiview without model
        vis_cam.remove_geometry(model, reset_bounding_box=False)
        snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
        if save_snapshots:
            plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidar_{lidar_ids[0]}_{lidar_ids[1]}_multiview_wo_model.png'), snapshot)

        # snapshot3: multiview without ray
        for i, _ in enumerate(lidar_ids):
            vis_cam.remove_geometry(rays[i], reset_bounding_box=False)
        arrow_len = 0.03 # default cylinder height is 5.0, so the scale factor is
        scale_factor = arrow_len / 5.0
        for i, lidar_id in enumerate(lidar_ids):
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.25, cone_radius=1.0)
            arrow.compute_vertex_normals() # this allows Phong shading!
            direction = np.array([0,0,0])-self.lidar_full[lidar_id,:3]
            direction /= np.linalg.norm(direction)
            R = rotation_matrix_from_vectors(np.array([0,0,1]), direction) # compute relative rotation from the default direction (0,0,1) to the lookat vector direction
            arrow.rotate(R)
            arrow.scale(scale=scale_factor, center=np.array([0,0,0]))
            arrow.translate(translation=self.lidar_full[lidar_id,:3] + direction * arrow_len/2, relative=False)
            arrow.paint_uniform_color(colors[i]) # orange
            vis_cam.add_geometry(arrow, reset_bounding_box=False)

        snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
        if save_snapshots:
            plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_lidar_{lidar_ids[0]}_{lidar_ids[1]}_multiview_wo_ray.png'), snapshot)

        if display:
            vis_cam.run()
        vis_cam.destroy_window()

    def vis_partial_clouds(self, display=False, save_snapshots=True):
        '''
        Visualize partial clouds.

        :param bool display flag to display Open3D windows
            - window display the clouds
        :param bool save_snapshots flag to save intermediate results
            - figure showing partial clouds as the views increase, a sequence of 'RRX_N_lidars_M_cloud.png' for M being the number of lidars in partial and full.
        '''
        vis_cam = o3d.visualization.Visualizer()
        vis_cam.create_window('LiDAR Clouds', width=1280, height=1280, visible=display)
        color = [0,0.1,0.6] # blue

        # draw model
        model = self.mesh
        model.scale(scale=1.0,center=np.array([0,0,0]))
        vis_cam.add_geometry(model, reset_bounding_box=True) # prevent viewpoint from changing automatically
        vis_cam.remove_geometry(model, reset_bounding_box=False)
        
        # look-at camera model
        # ref: https://ksimek.github.io/2012/08/22/extrinsic/
        view_control = vis_cam.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(np.array([-1,1,0.5])) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction

        # draw LiDAR locations and save snapshots
        j = 0
        cloud = np.empty((0,3), float)
        for i in self.lidar_partial + [self.N_lidar_full]:
            # accumulate lidar data
            while j < i:
                cloud = np.concatenate((cloud, self.rayhits[j]), axis=0)
                j += 1
            # draw partial cloud
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cloud) # from numpy to o3d format
            pc.paint_uniform_color(np.array(color)) 
            vis_cam.add_geometry(pc, reset_bounding_box=False)

            # save
            snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
            if save_snapshots:
                plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_cloud_lidars_{j}.png'), snapshot)
                # o3d.io.write_point_cloud(os.path.join(self.snapshot_path, f'{self.model_name}_cloud_lidars_{j}.ply'), pc)

            # reset cloud
            vis_cam.remove_geometry(pc, reset_bounding_box=False)
            
        if display:
            vis_cam.run()
        vis_cam.destroy_window()
    
    def vis_partial_clouds_w_permutation(self, permutation, display=False, save_snapshots=True):
        '''
        Visualize partial clouds.

        :param bool display flag to display Open3D windows
            - window display the clouds
        :param bool save_snapshots flag to save intermediate results
            - figure showing partial clouds as the views increase, a sequence of 'RRX_N_lidars_M_cloud.png' for M being the number of lidars in partial and full.
        '''
        vis_cam = o3d.visualization.Visualizer()
        vis_cam.create_window('LiDAR Clouds', width=1280, height=1280, visible=display)
        color = [0,0.1,0.6] # blue

        # draw model
        model = self.mesh
        model.scale(scale=1.0,center=np.array([0,0,0]))
        vis_cam.add_geometry(model, reset_bounding_box=True) # prevent viewpoint from changing automatically
        vis_cam.remove_geometry(model, reset_bounding_box=False)
        
        # look-at camera model
        # ref: https://ksimek.github.io/2012/08/22/extrinsic/
        view_control = vis_cam.get_view_control()
        view_control.set_lookat(np.array([0,0,0])) # lookat is the target focus
        view_control.set_front(np.array([-1,1,0.5])) # front is likely just the camera position, i.e. camera position - lookat position
        view_control.set_up(np.array([0,0,1])) # up is the upright direction

        # draw LiDAR locations and save snapshots
        j = 0
        cloud = np.empty((0,3), float)
        for i in self.lidar_partial + [self.N_lidar_full]:
            # accumulate lidar data
            while j < i:
                cloud = np.concatenate((cloud, self.rayhits[j]), axis=0)
                j += 1
            # draw partial cloud
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cloud) # from numpy to o3d format
            pc.paint_uniform_color(np.array(color)) 
            vis_cam.add_geometry(pc, reset_bounding_box=False)

            # save
            snapshot = np.asarray(vis_cam.capture_screen_float_buffer(do_render=True))
            if save_snapshots:
                plt.imsave(os.path.join(self.snapshot_path, f'{self.model_name}_cloud_p{permutation}_lidars_{j}.png'), snapshot)
                # o3d.io.write_point_cloud(os.path.join(self.snapshot_path, f'{self.model_name}_cloud_lidars_{j}.ply'), pc)

            # reset cloud
            vis_cam.remove_geometry(pc, reset_bounding_box=False)
            
        if display:
            vis_cam.run()
        vis_cam.destroy_window()

if __name__ == '__main__':
    root_path = 'H:/RockScan'
    rock_category = 'RR4'
    start_folder = 1
    end_folder = 36
    result_path = "H:/git_symphonylyh/3D/Completion3D/SnowflakeNet/datasets/rocks3d/train"
    snapshot_path = "H:/git_symphonylyh/3D/Completion3D/SnowflakeNet/datasets/rocks3d/vis"
    # hyper-parameters
    N_orientations = 16 # number of different particle orientation views
    N_lidar_full = 16 # number of full LiDARs
    N_lidar_partial = list( range(int(N_lidar_full*0.2), int(N_lidar_full*0.6)+1) ) # use the first N (10%-60%) of the full LiDARs
    ring_spacing = 0.002
    arc_spacing = 0.002
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

        ### Raycasting to get hits from all lidars
        r.raycasting_full(N_ring=N_ring, ring_spacing=ring_spacing, arc_spacing=arc_spacing)

        ### Visualization
        # this is done once per model, i.e., only save snapshots for the first orientation, otherwise it's too many repeated snapshots
        # (1) visualize lidar locations for each lidar set (partial lidars in orange)
        r.vis_lidar_locations(display=False, save_snapshots=True)
        # (2) visualize lidar rays
        r.vis_lidar_rays(display=False, save_snapshots=True)
        # (3) visualize multi-view lidar rays
        r.vis_lidar_multi_views(display=False, save_snapshots=True)
        # (4) visualize partial (and complete) clouds
        r.vis_partial_clouds(display=False, save_snapshots=True)

        # repeat
        # for each model: 
        #   init LiDAR sets
        #   for each orientation:
        #       raycast one full LiDAR set
        #       for different # of selected partial LiDARs:
        #           raycast several partial LiDARs sets
        orientations = r.compute_orientations(N=N_orientations)
        print(f'Each model is permutated by {len(orientations)} times, so each model has {len(r.lidar_partial)}x{len(orientations)}={len(r.lidar_partial)*len(orientations)} partial-gt pairs.')
        for o in range(N_orientations):
            # raycasting all
            r.raycasting_full(N_ring=N_ring, ring_spacing=ring_spacing, arc_spacing=arc_spacing)

            # debug save snapshot for multiple permutations of a certain rock
            # if folderID == start_folder:
            #     r.vis_partial_clouds_w_permutation(permutation=o, display=False, save_snapshots=True)

            # save partial and complete clouds
            r.save_clouds(permutation=o, partial_points=num_points_per_partial, complete_points=num_points_per_gt)
            
            # permute the particle orientation & repeat
            old_orientation = np.array([0,0,1]) if o == 0 else orientations[o-1] # assume the initial orientation is (0,0,1), doesn't matter since we rotate many times
            new_orientation = orientations[o]
            r.permute_orientation(old_orientation, new_orientation)
            
    
    