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

On Windows:

```bash
// create an empty folder named snowflake
conda create --prefix H:\envs\snowflake python=3.7
conda activate H:\envs\snowflake

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch # follow: https://pytorch.org/get-started/previous-versions/, our windows desktop is cuda 10.2
H:\envs\snowflake\Scripts\pip.exe install -r requirements.txt


```

On Ubuntu:

```bash
conda create --name pytorch180cudnn111 python=3.7 # this is done in pointgroup installation
conda activate pytorch180cudnn111
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
# pytorch 1.8.0 and cudnn 11.1 seems to work best on RTX 3090 (note Ampere 3090 only supports CUDA >= 11)
```

