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



On Windows:

```bash
conda create --prefix H:\envs\snowflake python=3.7
conda activate H:\envs\snowflake
H:\envs\grnet\Scripts\pip.exe install -r requirements.txt
```

