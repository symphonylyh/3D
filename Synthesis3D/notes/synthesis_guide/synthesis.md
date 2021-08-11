Date: 05/2021

# Synthesis3D Guide

[TOC]

This section is after the [3D reconstruction section](../../../Reconstruction3D/notes/agisoft_metashape_guide/agisoft_metashape_guide.md).

## Pre-processing in MeshLab and Blender

As the preparation steps for 3D synthetic data generation in Unity, the following major tasks are necessary:

* Step 1: Export Reconstructed Rock Models (Metashape/Laser Scan)
* Step 2: Mesh Re-centering/Simplification/Downsampling (MeshLab)
* Step 3: Fabricate Level-of-Detail (LOD) Model (Blender)
* Step 4: Import to Unity

### Export Reconstructed Rock Models from Metashape/Laser Scan

#### Metashape

RR3 and RR4 rocks are reconstructed using SfM. These are models with texture.

Which data format is the most suitable? The Autodesk FBX file format is a popular 3D data interchange format utilized between 3D editors and graphics engines. Unity generally takes .fbx or .obj as mesh model. MeshLab can import but cannot export .fbx. .fbx model is more compressed than .obj model. As a result, I choose .fbx as the protocol format between Metashape & Meshlab as well as between Blender & Unity, and choose .obj as the protocol format between MeshLab & Blender. 

To export model from Metashape with texture, File -- Export -- Export Model, save colors and normals, and export texture as a separate file (jpeg was found to have the smallest size). If "Embed texture" is checked, there will be an extra `.fbm` folder. I found this is not necessary, Unity/MeshLab/Blender can automatically match the texture file with the model. 

![image-20210425000652188](figs/image-20210425000652188.png)

As a result, for each rock we have a raw `.fbx` model and a `.jpg` texture.

For automated export from Metashape, see [guide](../../../Reconstruction3D/notes/agisoft_metashape_guide/agisoft_metashape_guide.md) and [script](../../../Reconstruction3D/metashape-workflow/metashape_batch_export.py).

#### Laser scan

Ballast rocks are reconstructed using laser scanner. These are textureless models. 

They have reddish, grayish, and white colors. So I use SfM to reconstruction a few rocks of each color, and map the texture to all laser scanned models.

I found the polygon mesh generated from the laser scanner sometimes contain irregular faces, so before calculating the surface area and volume, I apply Taubin smooth to the mesh.

Then the mesh needs to be parameterization before assigning texture. I use Parameterization: Flat Plane.

Finally, set the texture map for the mesh and export as PLY model.

The above steps are automated by this Pymeshlab [script]().

### Mesh Re-centering/Simplification/Downsampling in MeshLab

First, after reconstruction, the model is usually with certain offset. It is better to re-center the model to its centroid. This can be done in MeshLab by Filters -- Normals, Curvatures, and Orientation --  Transform: Translate, center, set origin -- Center on Scene/Layer BBox -- Freeze Matrix, as described [here](https://revthat.com/updating-origin-meshes-meshlab/). 

![image-20210503141421350](figs/image-20210503141421350.png)

Second, the model needs to be downsampled to be smoothly rendered in Unity. Filters -- Remeshing, Simplification, Reconstruction -- Simplification: Quadric Edge Collapse Decimation (with Texture). Since we want 3 LOD levels in Unity, we may target at 2000, 1000, 500 faces.

![image-20210425014918598](figs/image-20210425014918598.png)

This is called decimating mesh in graphics.

In addition, geometry properties can be computed in Meshlab, Filters -- Quality Measures computations -- Compute Geometric Measures. It computes bounding box, surface area, barycenter, volume, etc. Can check source code, the volume calculation is based on "Mirtich, B. (1996). Fast and accurate computation of polyhedral mass properties. *Journal of graphics tools*, *1*(2), 31-50."

To automate everything, three options are available:

* Latest release [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) library with the 2020.12 version! It offers the most efficient Python API of MeshLab. And it's official. This option is the best solution.
* Record the script from Filters -- show current filter script -- save/load script. This is ok, but no longer supported after the release of PyMeshLab. This option is not used.
* Python library MeshLabXML. This one provides more flexible interfaces, however, it is not official API so it may not work with the latest version of MeshLab. This option is not used.

The batch processing Python script is [here](../../synthesis-workflow/meshlab_LOD_generation.py).

Another more research-oriented mesh simplification approach would be using [Instant Meshes](https://github.com/wjakob/instant-meshes). The usage can be found [here](https://blender.stackexchange.com/a/108322).

### Fabricate Level-of-Detail (LOD) Model in Blender

To scripting Blender, two options are available:

* run script in command line. Details description [here](https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html). First locate the `blender.exe`, then call it by command `blender.exe --background --python test.py [argv...]` then everything will run without GUI.

* `pip install bpy` [link](https://pypi.org/project/bpy/) is available but not officially supported. 

The batch processing Python script is [here](../../synthesis-workflow/blender_entry.py) and [here](../../synthesis-workflow/blender_LOD_merge.py). Note that there is a tricky thing about the scale. Since FBX use 1 unit = 1cm by default, we need to set the scene scale to 0.01 (i.e. 1 Blender unit=0.01m=1cm). The scale is set in script  [here](../../synthesis-workflow/blender_LOD_merge.py). If scene scale is 1.0 (1 Blender unit=1m), when importing obj (in cm unit, e.g. a 10cm rock), the object will only take 0.1 Blender unit. And when exporting as fbx, it will be 0.1 Blender unit=0.1cm. Therefore it's downscaled by a factor of 100...

To collect all the LOD models to be used as Unity assets, use this [script](../../synthesis-workflow/gather_LODs.py).

Blender tips:

* Alt + mid mouse: auto focus on mouse
* Show texture with model: on the upper right corner, switch to solid mode, click the drop down, and choose texture in color.

![image-20210425003648458](figs/image-20210425003648458.png)

### Import to Unity

Now the `.fbx` can be imported to Unity along with the corresponding texture `.jpg` file. But what's imported is still the "model" instead of "prefab". Prefab contains Unity components, such as "RigidBody", "Mesh Collider" that we care most.

For this, open any empty scene and drag the models into it (**Important: set the scale factor to 100 in the model import setting**). Select all the models and go to menu `Synthesis3D > FBX to Prefab`. The editor script [`FBX2Prefabs.cs`](../../synthesis-workflow/synthesis3D/Assets/Synthesis3D/Editor/FBX2Prefabs.cs) will be executed and saved models as prefabs.

The editor script:

* Set LOD transition levels
* Add Rigidbody. Note that the collision detection mode can be set as Discrete (may miss high speed object) or Continuous Dynamic (accurate but costly)
* Add Mesh collider. Set the collider mesh as the coarsest LOD level.

Since later on we need ray casting to read the texture pixels, remember to enable read/write for texture files. Select all textures -- Import Settings -- Advanced -- tick "Read/Write Enabled".

## Synthetic Data Generation in Unity

After the steps above, the synthetic data generation in Unity requires the following major tasks:

* Step 1: Instantiation of Rock Models
* Step 2: Stockpile Generation by Gravity Falling
* Step 3: Multi-View Image Data Generation by Programming Camera Movement
* Step 4: 3D Point Cloud Data Generation by Ray Casting
* Step 5: Export Synthetic Data

### 3D Point Cloud Data Generation by Ray Casting

BoxCast and SphereCast are good options, but user has no control over the ray density. Instead, I should create my own point cloud cast functionality.

Important note:

Retrieve Cartesian coordinates is straightforward. But ray cast hit will only return the correct texture coordinate if:

* The gameobject has a MeshCollider. Others like SphereCollider or BoxCollider won't work.
* The MeshCollider is non-convex. This is because the texture of a convex collider is undefined.
* The texture file is Read/Write Enabled in the import settings.

This leads to a conflict: during gravity falling, the rigidbody must have convex collider for correct collision detection; but during ray casting, the collider must be non-convex for correct texture coordinate. The final solution is to change the properties at different phases:

* During gravity falling, Rigidbody has `isKinematic=false,collisin `, MeshCollider has `convex=true`
* During ray casting, Rigidbody has `isKinematic=true`, MeshCollider has `convex=false`

Kinematics vs. Dynamics. Kinematics studies the motion without regard to forces that cause it; Dynamic studies the motions that result from forces. In Unity, kinematic means the object is controlled by setting the transform instead of interaction with forces.

### Export Synthetic Data

File types: Binary (compact, fast read speed) and ASCII.

* `XYZ`: ASCII
* `OBJ`: Binary & ASCII
* `PLY`: Binary & ASCII
* `LAS`: Binary. For LiDAR data.

`PLY` was found to be the most convenient option when we need to add per-vertex properties (e.g., in our case we want LiDAR ID, rock instance ID, etc.). See some writings in [blog](http://paulbourke.net/dataformats/ply/).

Note that Unity uses a left-handed coordinate system (X-forward, Y-up, Z-left) while other common software like Blender & Meshlab & Open3D all use a right-handed coordinate system (X-forward, Y-up, Z-right). To align this, we negate the Z value when exporting the PLY (including points and LiDAR positions). Later on we will subtract all XYZ coordinates by the minimum values such that the entire point cloud will be shifted to start from (0,0,0), and everything looks the same as in Unity.

Use `plot3d_example_unity.py` to visualize the data. `prepare_synthetic_data_h5.py` to convert PLY to h5 format.