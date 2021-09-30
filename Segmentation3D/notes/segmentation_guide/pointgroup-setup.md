Date: 06/2021

## PointGroup Guide

### Github Setup

* We want to fork pointgroup and get up to date with the original repo (although it doesn't seem to update since 2020)
* We want to add the pointgroup repo as a subdirectory in my big 3D/Segmentation3D repo
  * Fork the repo on github web, now this forked repo has a link `https://github.com/symphonylyh/PointGroup.git`
  * Go to the big 3D repo, add the forked repo as a subtree: `git subtree add --prefix Segmentation3D/PointGroup https://github.com/symphonylyh/PointGroup.git master --squash`. This will create a folder `3D/Segmentation3D/Points3D` which is the master branch of the torch-points3d repo. use `--squash` to merge as just one commit.
  * Commit it in Github Desktop or by `git push origin main`. Now this repo is added as a subrepo
  * say the local repo is A, the forked repo is B, the original repo is C. ABC are different repos.
  * General push just goes to A by `git pull/push origin main`; to update A <--> B, `git subtree push --prefix Segmentation3D/Points3D https://github.com/symphonylyh/torch-points3d.git master` and `git subtree pull --prefix Segmentation3D/Points3D https://github.com/symphonylyh/torch-points3d.git master --squash`; to update B <-- C, on github web, click "Fetch Upstream" --> "Fetch and merge"
* A subdirectory can also be extracted to be a separate repo by subtree. Check out that later.

### Setup

Note: pointgroup is based on spconv which is not supported on Windows. So the setup guide is linux only. Our linux machine has RTX 3090 so can only support CUDA >= 11. Fortunately, pointgroup code turns out to be compatible with CUDA 11.

```bash
conda create --name pytorch180cudnn111 python=3.7 
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
# pytorch 1.8.0 and cudnn 11.1 seems to work best on RTX 3090 (note Ampere 3090 only supports CUDA >= 11)

conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
/home/luojiayi/anaconda3/envs/pytorch180cudnn111/bin/pip install cmake>=3.13.2

cd lib/spconv
python setup.py bdist_wheel
```

Many errors occur when build the spconv library. We go to the [spconv rep](https://github.com/traveller59/spconv)o for answers.

* Error 1: `-- Could NOT find CUDNN (missing: CUDNN_LIBRARY_PATH CUDNN_INCLUDE_PATH) `. Since this is build from source instead of using cuda toolkit, we need to install cuDNN properly. Go to cuDNN [download page](https://developer.nvidia.com/cudnn), login with NV account, and download v8.0.5 for CUDA 11.1, cuDNN library for Linux (x86_64). Then follow Tar file installation [guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar). Basically it copies cudnn files to the local cuda folder. After this, the cudnn can be found
* Error 2: `error: no matching function for call to ‘torch::jit::RegisterOperators::RegisterOperators(const char [28], <unresolved overloaded function type>)’`. This is because pytorch >=1.4 has remove the jit namespace, so go to `lib/spconv/src/spconv/all.cc:20` and remove `::jit`

After these, spconv wheel is built successfully! Go to dist/ to pip it

```bash
cd dist
/home/luojiayi/anaconda3/envs/pytorch180cudnn111/bin/pip install spconv-1.0-cp37-cp37m-linux_x86_64.whl
```

Other dependencies are easier to handle

```bash
/home/luojiayi/anaconda3/envs/pytorch180cudnn111/bin/pip install plyfile tensorboardX pyyaml scipy
conda install -c anaconda mayavi # visualization

cd lib/pointgroup_ops
python setup.py develop

/home/luojiayi/anaconda3/envs/pytorch180cudnn111/bin/pip install open3d
```

Success!

### Dataset Preparation

For training with own data, the authors give some advice in this [issue](https://github.com/dvlab-research/PointGroup/issues/3), for example changing the input channel and dataloader etc.

First we need to understand how Scannetv2 dataset looks like:

* `[scene]_vh_clean_2.ply`: low-res RGB mesh. Points has (x,y,z,r,g,b,a). +Z is upright direction. RGB is 0-255
* `[scene]_vh_clean_2.labels.ply`: low-res label mesh. Points has (x,y,z,r,g,b,a,sem_label)
* `[scene]_vh_clean_2.0.010000.segs.json`: low-res segment/part index. Each object instance has several segments (parts, called over-segmentation), and each segment is assigned a unique ID called `segIndices` in this file (not necessarily starts from 0, but is unique per segment). And then, this file give the per-vertex segment index, e.g., adjacent points forming the segment has the same segment index.
* `[scene].aggregation.json`: low-res instance-level semantic segmentation. Each object contains a list of segIndices, i.e., the segments that form this object instance. So to get instance label for each point will need map between segs.json and aggregation.json

Then we need to understand how PointGroup pre-process the above files into a pth file. See [prepare_data_inst.py](../../PointGroup/dataset/scannetv2/prepare_data_inst.py).

* point (x,y,z) centered around the origin, i.e. subtract mean(x,y,z)
* color normalized to [-1,1] range
* collect per-point semantic label from `lables.ply`. Semantic label starts from 0 to nclass-1. Unknown classes have label of -100
* create a map between segment ID --> point indices belonging to this segment, from `segs.json`
* collect per-point instance label by merging the segments, from `aggregation.json`. Instance label starts from 0. Unknown points have label of -100
* save (coords, colors, sem_labels, instance_labels) into a pth file. For test split, no sem and ins labels are generated
* Different from GICN, the entire scene is passed instead of dividing into blocks

The above pre-processing is for training. For evaluation on validation, we should further use [prepare_data_inst_gttxt.py](../../PointGroup/dataset/scannetv2/prepare_data_inst_gttxt.py) that generate instance ground truth file under `/val_gt/[scene].txt`. PointGroup used a weird encoding `x00y` that x is semantic label, y is instance label and they do x*1000+y+1, assuming less than 1000 instances. And instead of -100, now unannotated semantic & instance labels is 0. Semantic labels is nyu40 labels instead of 0 to nclass-1.

```bash
cd dataset/scannetv2

python prepare_data_inst.py --data_split train # val, test as well, into .pth files
python prepare_data_inst_gttxt.py # by default work on val set
```

Finally we can process our own dataset. Since we already generate all labels in a ply file, it's easier in our case to generate such pth file. See [prepare_data_color.py](../../PointGroup/dataset/primitives3d/prepare_data_color.py) and [prepare_data_colorless.py](../../PointGroup/dataset/primitives3d_colorless/prepare_data_colorless.py) depending on whether the data is color or colorless. Note that our data is double, we need to convert to float since pointgroup cuda operation requires torch float tensor, so we need cast to np.float32 during data pre-processing.

```bash
cd dataset/primitives3d

python prepare_data_color.py --data_split train # val, test as well, into .pth files
```

### Dataloader

The dataloader is defined in  [scannetv2_inst.py](../../PointGroup/data/scannetv2_inst.py). For each pth, `trainMerge` is called as the transform, which does:

* Data augmentation: initialize a 3x3 transformation matrix as identity E. jitter - add ~0.1*N(0,1) to E. flip: make E[0,0]=-1 or 1, i.e. flip x randomly. rotate: apply a [0,2pi] rotation to the points. **Train does all three. Val and Test do not jitter**. And the performance varies a little, see this [issue](https://github.com/dvlab-research/PointGroup/issues/24#issue-724009279).
* Data augmentation: elastic distortion. Elastic distortion has a granularity and magnitude, see points3D [doc](https://torch-points3d.readthedocs.io/en/latest/src/api/transforms.html#torch_points3d.core.data_transform.ElasticDistortion). The description of elastic distortion can date back to this [paper](https://cognitivemedium.com/assets/rmnist/Simard.pdf). In [Minkowski Engine paper](https://arxiv.org/pdf/1904.08755.pdf), they mentioned "Since the dataset is purely synthetic, we added various noise to the input point clouds to simulate noisy observations. We used elastic distortion, Gaussian noise, and chromatic shift in the color for the noisy 4D Synthia experiments". Our synthetic dataset may apply these too. **Train distort, Val and Test don't distort.**
* Crop: if the number of points exceed the threshold, partial scene is cropped. **Train and Val crop, Test doesn't crop**.
* Merge: one data sample is a scene, and the batch will merge multiple scenes together. Therefore, in each forward pass the dimension of the batch may differ. The instance label accumulates. For example, scene A and B both have instances 0-10, then scene B instance label will become 11-20. `batch_offsets` is the cumulative batch start point ID, `locs` (N, 4) is (batch_id, x,y,z) for all points.
* Voxelize: Pointgroup divide with voxel size 0.02m, i.e. multiple the coords by 50. The coords are passed to CUDA operations, see [`pointgroup_ops.py`](../../PointGroup/lib/pointgroup_ops/functions/pointgroup_ops.py) for the API. This is a wrapper around the compiled CUDA code in PG_OP. The real CUDA code is under `lib/pointgroup_ops/src/voxelize/voxelize.cu`. To further understand the code, in `voxelize.cpp` `SparseGrids` is a Google spatial hash structure that maps a voxel coord (x,y,z) to a voxel index (linearized ID). In `voxelize_inputmap()`, points hash into voxels. The number of active voxels (M, voxels that contain at least one point) and the list of points inside each voxel are recorded. SpConv allows different modes: mode 0=guaranteed unique 1=last item(overwrite) 2=first item(keep) 3=sum, 4=mean. PointGroup uses mode 4=mean. It seems this mean doesn't mean average, it still`input_map or p2v_map` maps point ID to voxel ID (N-to-1 map), `output_map or v2p_map` maps back (1-to-N) map. **Train, Val, Test all does this.**

### Train

Pointgroup repo has command to train on Scannet v2, we need to download partial Scannet. We can find the download script `/torch-point3d/scripts/datasets/download-scannet.py`. And use `python download-scannet.py -o . --type ... --id scene0000_00`. The meaning of each file can be found [here](http://www.scan-net.org/ScanNet/). Then prepare the dataset structure as Pointgroup requires and generate the [scene]_inst_nostuff.pth for training. Remember if we only use one or two scenes in the train dataset, we need to change the batch size to 1 or 2 in `pointgroup/config/pointgroup_run1_scannet.yaml` otherwise the dataloader is empty. Also change the epoch in the yaml file to a small number.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_run1_scannet.yaml # models will be under /exp

python train.py --config config/pointgroup_default_primitives3d.yaml # own dataset

tensorboard --logdir=./exp # create tensorboard session
```

The main function is `model/pointgroup/pointgroup.py: model_fn_decorator()`. This calls the forward() in PointGroup nn.Module. The backbone conv, clustering, and scorenet can be found here:

* Backbone: One question that bothers me is that data batch is multiple point clouds stacked together. Then pass through spconv. But spconv will compute local features therefore it actually computes over a mix-up point cloud. Does such features useful? Or should I always use batch_size=1? 
* Clustering: Although for the clustering part it does distinguish between scenes (see `lib/pointgroup_ops/src/bfs_cluster/bfs_cluster.cu: ballquery_batch_p_cuda_()`). For each point (o_x,o_y,o_z), it selects all the adjacent points (less than 1000) within certain radius. These adjacent points are pre-screened to be in the same scene (see start & end variables). After selecting the points within radius, it generates instance proposals (points with the same semantic labels as the current point), see `lib/pointgroup_ops/src/bfs_cluster/bfs_cluster.cu: bfs_cluster()`. The selection - proposal step is applied to both shifted coords and original coords. The returned `idx_shift` and `idx` are serialized point ID array of ball1 points, ball 2 points, ... `start_len_shift` and `start_len` are range pointers telling `start_len[i, 0] ~ start_len[i,1]` is one ball. `proposals_idx` is (NInstancePoints, 2) where each entry is (instance ID, global point ID), e.g. if point 11 belongs to instance No. 3, the entry in `proposals_idx` is (3,11). `proposal_offsets` helps to quickly find the start pointer of certain instance, so it's a cumsum of NInstancePoints. The proposals based on original coords and shifted coords are concatenated so the first column in `proposals_idx_shift[:,0]` should be offset by the total NInstance of `proposals_idx`.
* ScoreNet: proposals are voxelized again before scoring. Note that NMS is not applied during training. Loss are calculated based on the raw proposals instead of NMS results. NMS only applies during testing (see `test.py`)

### Test and Inference

After downloading the pretrained model

```bash
python test.py --config config/pointgroup_run1_scannet.yaml 
# change split: val, eval: True, this will, test_epoch to whatever epoch you want to test
# change split: test, eval: False, save_instance: True on test set

python test_synthetic.py --config config/pointgroup_default_primitives3d_colorless.yaml 

cd util
python visualize.py --data_root=../dataset/scannetv2 --result_root=../exp/scannetv2/pointgroup/pointgroup_default_scannet/result/epoch384_nmst0.3_scoret0.09_npointt100 --room_name=scene0000_00 --room_split=test --task=instance_pred
```

### Notes

* this [push history](https://github.com/symphonylyh/3D/commit/b22dbd47f4a17e25f44c736ce74ffd9a5e5154fd#diff-50ee2a3bad7b3c5ee26e47b47885e1b680610e1afe7c6b7ac3432f32bd43fff2) records a change in pointgroup.py that originally ignores all semantic predictions <=1, but we have semantic starting from 0, so 0 and 1 are both object categories.

