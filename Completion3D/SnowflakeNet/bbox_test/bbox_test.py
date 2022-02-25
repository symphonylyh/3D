'''
Bounding box of a point set or mesh is important in the calcuation of shortest/intermediate/longest dimensions and FER3D. At some point I noticed that the bbox from pymeshlab's computer_geometric_measures() and open3d's bbox are different!

The problem is: pymeshlab only computes an axis-aligned bbox, but axis-aligned bbox changes every time when the particle rotates. The right way should be calculating the oriented bbox by open3d's OrientedBoundingBox.create_from_points(), such that it's constant at any orientation
'''

import pymeshlab as ml
import open3d as o3d
import numpy as np

test_model = 'RR3_14_l6_p013_gt.ply'

ms = ml.MeshSet()
ms.load_new_mesh(test_model)
pcd1 = ms.mesh(0).vertex_matrix()

pc1 = o3d.geometry.PointCloud()
pc1.points = o3d.utility.Vector3dVector(pcd1)
# print(pcd1[:10,:])
bbox1_o3d = o3d.geometry.OrientedBoundingBox.create_from_points(pc1.points)
bbox1_dim = bbox1_o3d.extent
bbox1_dim = np.sort(bbox1_dim)
print("Open3d bbox dim before rotation: ", bbox1_dim, ", FER3D", bbox1_dim[2]/bbox1_dim[0])

pc1.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi/4, np.pi/2, -np.pi/3)))

pcd2 = np.asarray(pc1.points)
# print(pcd2[:10,:])
pc2 = o3d.geometry.PointCloud()
pc2.points = o3d.utility.Vector3dVector(pcd2)
bbox2_o3d = o3d.geometry.OrientedBoundingBox.create_from_points(pc2.points)
bbox2_dim = bbox2_o3d.extent
bbox2_dim = np.sort(bbox2_dim)
print("Open3d bbox dim after rotation: ", bbox2_dim, ", FER3D", bbox2_dim[2]/bbox2_dim[0])

m1 = ml.Mesh(pcd1)
ms1 = ml.MeshSet()
ms1.add_mesh(m1, 'pcd1')
measures1 = ms1.compute_geometric_measures()
bbox1_ml = measures1['bbox']
bbox1_dim = (bbox1_ml.dim_x(), bbox1_ml.dim_y(), bbox1_ml.dim_z())
bbox1_dim = np.sort(bbox1_dim)
print("Meshlab bbox dim before rotation: ", bbox1_dim, ", FER3D", bbox1_dim[2]/bbox1_dim[0])

m2 = ml.Mesh(pcd2)
ms2 = ml.MeshSet()
ms2.add_mesh(m2, 'pcd1')
measures2 = ms2.compute_geometric_measures()
bbox2_ml = measures2['bbox']
bbox2_dim = (bbox2_ml.dim_x(), bbox2_ml.dim_y(), bbox2_ml.dim_z())
bbox2_dim = np.sort(bbox2_dim)
print("Meshlab bbox dim after rotation: ", bbox2_dim, ", FER3D", bbox2_dim[2]/bbox2_dim[0])
