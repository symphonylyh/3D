[TOC]

As the preparation steps for 3D synthetic data generation in Unity, the following major tasks are necessary:

* Step 1: Export Reconstructed Rock Models from Metashape
* Step 2: Mesh Simplification/Downsampling in MeshLab
* Step 3: Fabricate Level-of-Detail (LOD) Model in Blender
* Step 4: Import to Unity

After the steps above, the synthetic data generation in Unity requires the following major tasks:

* Step 1: Instantiation of Rock Models
* Step 2: Stockpile Generation by Gravity Falling
* Step 3: Multi-View Image Data Generation by Programming Camera Movement
* Step 4: 3D Point Cloud Data Generation by Ray Casting
* Step 5: Export Synthetic Data



## Export Reconstructed Rock Models from Metashape

Which data format is the most suitable? The Autodesk FBX file format is a popular 3D data interchange format utilized between 3D editors and graphics engines. Unity generally takes .fbx or .obj as mesh model, so I choose .fbx as the protocol format between MeshLab/Blender and Unity. 

## Mesh Simplification/Downsampling in MeshLab

This is called decimating mesh in graphics.

## Fabricate Level-of-Detail (LOD) Model in Blender

## Import to Unity