Date: 08/2021

## Overview

3D shape completion has some similarities with the problem of texture mapping or mesh parameterization.

[surface parameterization survey](https://graphics.stanford.edu/courses/cs468-05-fall/Papers/param-survey.pdf)



Blender has UV unwrapping methods.

Promising networks are:

* [PCN](https://github.com/wentaoyuan/pcn) (Point Completion Network, 3DV 2018), tensorflow
* [TopNet](https://github.com/lynetcha/completion3d) (CVPR 2019), tensorflow & pytorch

* [PFNet](https://github.com/zztianzz/PF-Net-Point-Fractal-Network) (Point Fractal Network, CVPR 2020), pytorch
* [CRN](https://github.com/xiaogangw/cascaded-point-completion) (Cascaded Refinement Network, CVPR 2020), tensorflow
* [MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion) (Morphing and Sampling Network, AAAI 2020), pytorch
* [GRNet](https://github.com/hzxie/GRNet) (Gridding Residual Network, ECCV 2020), pytorch
* [NSFA](https://github.com/XLechter/Detail-Preserved-Point-Cloud-Completion-via-SFA) (Detailed Preserved Point Cloud Completion via Separated Feature Aggregation, ECCV 2020), tensorflow 
* [PMP-Net](https://github.com/diviswen/PMP-Net) (Point Moving Paths Network, CVPR 2021), pytorch.
* [ShapeInversion](https://github.com/junzhezhang/shape-inversion) (CVPR 2021), pytorch. ShapeInversion is unsupervised method, so the performance is a little worse.
* [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet) (ICCV 2021), pytorch.

An approximate performance ranking (partially from shape inversion paper) is: TopNet < PCN < MSN , CRN, GRNet < PMP-Net, NSFA < SnowflakeNet .

GRNet, **PMP-Net, SnowflakeNet** have similar folder setup steps. PMP-Net and SnowflakeNet are actually from almost same authors.



Fork repo:

```bash
// fork on github
git subtree add --prefix Completion3D/SnowflakeNet https://github.com/symphonylyh/SnowflakeNet.git main --squash
// commit to github
```

Update pip requirements:

* pprint is built-in python library, remove it from pip list
* although pytorch > 1.4.0 is required, we force to use 1.8.0. And conda is preferred, so we remove it from pip list. pip install torch has many issues with missing headers, etc.
* our open3D visualization uses 0.13, so remove the 0.9.0.0

On Windows:

```bash
// create an empty folder named snowflake
conda create --prefix H:\envs\snowflake python=3.7
conda activate H:\envs\snowflake

# clean disk space
conda clean --all --dry-run # show prunable packages. remove --dry-run to actually clean up
H:\envs\snowflake\Scripts\pip.exe cache purge
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch # follow: https://pytorch.org/get-started/previous-versions/, our windows desktop is cuda 10.2
H:\envs\snowflake\Scripts\pip.exe install -r requirements.txt
H:\envs\snowflake\Scripts\pip.exe install pymeshlab plyfile pandas openpyxl
```

On Ubuntu:

```bash
conda create --name pytorch180cudnn111 python=3.7 # this is done in pointgroup installation
conda activate pytorch180cudnn111

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
# pytorch 1.8.0 and cudnn 11.1 seems to work best on RTX 3090 (note Ampere 3090 only supports CUDA >= 11)
/home/luojiayi/anaconda3/envs/pytorch180cudnn111/bin/pip install -r requirements.txt
/home/luojiayi/anaconda3/envs/pytorch180cudnn111/bin/pip install pymeshlab plyfile
```

```bash
cd pointnet2_ops_lib
python setup.py install

cd ..

cd Chamfer3D
python setup.py install
```



### SnowflakeNet Setup Guide

The Completion3D and ShapeNet datasets store the metadata in a JSON file first：

```bash
[
	{
        "taxonomy_id": "02691156",
        "taxonomy_name": "airplane",
        "test": [],
        "train": [
             "ec531add757ad0fa9a51282fb89c35c1",
             "21bf3d8201e3e41f93358ca8580664d1",
             ...
        ],
        "val": [],
    },
    {
        "taxonomy_id": "02933112",
        "taxonomy_name": "cabinet",
        "test": [],
        "train": [],
        "val": [],
    },
    ...
]
```

Taxonomy ID is a unique ID for an object category, and name. The string in the split is the file name

### Dataset preparation

* [`collect_complete_models.py`](../SnowflakeNet/datasets/collect_complete_models.py): collect all ground-truth models and re-sampling to 4096 points and save as h5 format
* [`stockpile2individual.py`](../SnowflakeNet/datasets/stockpile2individual.py): from the labeled synthetic stockpile, save each instance (partial cloud) as h5 format and associate with their ground-truth prototype h5 from the last step

### Train

Finally I prepared my dataset similar to ShapeNet (2048 input & 16384 output) so the model architecture should be same as the PCN file with up_factor=[4,8]. Change this in train/test/inference_rocks3d.py. But my dataset format is h5 so I need to use the same dataloader as Completion3D. So in data_loaders.py, remember to add the mapping at the end of the file.

```bash
python main_rocks3d.py
python main_rocks3d.py --test or --inference
```

### Architecture

**Encoder Part - Feature Extractor: PointNet++ & PointTransformer**

* 3 layers of PointNet++ Set Abstraction (SA) module to aggregate point features from local to global, interleaved by 2 layers of point transformer to incorporate local shape context. Point features are extracted by point transformer. 
* The input can be arbitrary number of points, they will first be re-sampling to a fixed input points (512) in the PointNet SA module by sample_and_group_knn(), using the farthest_point_sampling (FPS) in PointNet++. Then each point will search feature in a local regions of (16) neighboring points. Finally, go through MLP layers the entire partial cloud will be extracted as one global feature vector of size 1xC (=512).

**Footnote: what is farthest point sampling (FPS) algorithm?**

A sampling algorithm that can describe shape efficiently.

* Select an initial point set S={p0}, compute a distance array D that stores the distance from all other points to p0. The initial point could be randomly selected (but unstable), or more often chosen as the point that is farthest from the centroid of cloud (deterministic).
* pick p1 with the max distance D[i], add S={p0,p1}.
* compute distance from all points to p1, if d(pi,p1) < D[i], update D[i]. From this, the D array maintains the minimum distance from all points to the current set S.
* pick p2 with the max distance D[i], add S={p0,p1,p2}. Repeat until we sample target N points.
* The distance metric could be Euclidean for point cloud or geodesic for mesh.

**Decoder Part - Seed Generation (Coarse-Grained Decoder): Deconvolution & MLP**

* From the feature vector, generate a coarse complete cloud of 256 points
* do 1D deconvolution (transposed convolution) with a large receptive field (128) to get another C'=128 local feature vector per each of the 256 coarse point. Then, tile the C=512 global feature vector for each coarse point and concatenate with the deconved local feature, we have Nc x (C+C')=256x(512+128) extended features. **Note: the so-called point-wise splitting operation is essentially just this per-point deconvolution operation!**
* Pass the feature through MLPs and Conv1d, generate Nc=256 coarse points as the seeds, the variable `pcdc`.
* Concatenate the Nc=256 seeds (coarse complete clouds) with the N=2048 input partial clouds and do FPS to get N0=512 sparse cloud P0 for the next upsampling steps.

**Decoder Part - Upsampling (Fine-Grained Decoder): Snowflake Point Deconvolution (SPD) & Skip Transformer (ST)**

* SPD aims to upsample the points by splitting each parent point into multiple child points, which is done by first duplicating the parent points and then adding variations to the duplicates. Previous folding-based methods samples a same 2D grid around each parent point, which ignores the local shape characteristics around the parent point. SPD designs the point-wise splitting operation that fully leverages the local geometric information (features) around the parent point and generates the child points.
* Skip transformer is the key design. So the major contribution of SnowflakeNet is in the decoder process. Suppose the upsampling factor is r, the parent points are first duplicated with r copies. Each point is then passed through a ST and point splitting module and MLPs to get per-point displacement feature vectors K. Then a MLP compute the point coordinate shift deltaP (this is similar to pointgroup! maybe can utilize this ST structure too?). By adding the shift to the duplicate coordinates we get the upsampled points.
* ST uses the PointNet features (i.e. some MLP and Conv1d layers) as query Q, generates the shape context feature H, and further deconvolute (i.e. PS) to get the internal displacement features as key K. Q and H are of the dimension of the (i-1)-th cloud, and K is of the upsampled dimension of i-th cloud. Per-point q and k vectors are concatenated to be the value vector v, and Attention vector a is estimated based q and k. The attention vector denotes how much attention the shape context feature pays to each of the value vector v.
* It's important that displacement feature K from last SPD K_{i-1} is considered in the current i-th SPD step. This allows the shape context to propagate along the sequential upsampling process. In summary, skip transformer learn and refine the spatial shape context between the parent points and child points. "skip" represents the connection between the displacement feature K from previous SPD layer to the current layer.
* When upsampling factor r=1, it means to rearrange the seed points. Usually the upsampling sequence starts with a r=1, i.e. [1,2,2] or [1,4,8].

**Loss: Chamfer distance**

* The ground-truth cloud is down-sampled to the same density of Pc, P1, P2, P3, and Chamfer distance is used to calculated the loss.
* Another loss is preservation loss or partial matching loss. This is just a single-side Chamfer distance, i.e. S1 match to S2.

**Footnote: what is Chamfer distance?**

* measure the discrepancy between two point sets, S1 and S2.


$$
d_{CD, L_2}(S_1,S_2)=\frac{1}{N_1}\sum_{x\in S_1}\min_{y\in S_2} \lVert (x-y)^2 \rVert + \frac{1}{N_2}\sum_{y\in S_2}\min_{x\in S_1} \lVert (y-x)^2 \rVert
\\
d_{CD, L_1}(S_1,S_2)=\frac{1}{N_1}\sum_{x\in S_1}\min_{y\in S_2} \lVert x-y \rVert + \frac{1}{N_2}\sum_{y\in S_2}\min_{x\in S_1} \lVert y-x) \rVert
$$


### Questions

* 
