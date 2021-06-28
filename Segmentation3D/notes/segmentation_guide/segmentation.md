Date: 06/2021

# Segmentation3D Guide

[TOC]

This section is after the [3D synthesis section](../../../Synthesis3D/notes/synthesis_guide/synthesis.md).

## Deep Learning with 3D Scenes

2D images are represented in pixel grids, thus can be naturally handled by convolutional neural networks.

## Popular Approaches for Instance Segmentation

Good surveys:

* 2021 [arXiv](https://arxiv.org/abs/2103.05423)

Good collection of papers and codes:

* [page 1](https://github.com/Yochengliu/awesome-point-cloud-analysis)
* [page 2](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2021) for more recent papers

### Voxel-based Methods

This approach works on volumetric data where the 3D space is voxelized as voxel grids and each voxel has state either occupied or empty. Object in this representation is an entity with solid interior instead of a surface shell.

Examples:

* MTML, ICCV 2019 [[webpage](https://sites.google.com/view/3d-instance-mtml), [ICCV paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf), [tensorflow](https://github.com/lahoud/MTML)]. learns two sets of feature embeddings, including the feature embedding unique to every instance and the direction embedding that orients the instance center, which provides a stronger grouping force
* MASC, 2019 [[arXiv](https://arxiv.org/abs/1902.04478), [pytorch](https://github.com/art-programmer/MASC)].  predict the similarity embedding between neighboring points at multiple scales and semantic topology
* PanopticFusion, 2019, [[arXiv](https://arxiv.org/abs/1903.01177)].  predicts pixel-wise instance labels by 2D instance segmentation network Mask R-CNN for RGB frames and integrates the learned labels into 3D volumes
* OccuSeg, CVPR 2020, [[CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_OccuSeg_Occupancy-Aware_3D_Instance_Segmentation_CVPR_2020_paper.pdf)]

### Mesh-based Methods

But most current research focuses on point-cloud based methods that directly work on raw point cloud data without changing to voxel or mesh representation. The following approaches are point-cloud based methods.

### Point-based Methods

Examples:

* PointNet, PointNet++ for 3D feature extraction

### Proposal-based Approach

This approach relies on object box proposal or a pre-defined set of anchors to first localize object. Then it usually requires post-processing for refinement such as non-maximum suppression.

Pros:

Cons:

"Detection-based" means object bbox is explicitly generated during the approach, otherwise it's detection-free but still proposal-based.

Examples:

* [Detection-based] 3D-SIS, CVPR 2019, [[CVPR paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_3D-SIS_3D_Semantic_Instance_Segmentation_of_RGB-D_Scans_CVPR_2019_paper.pdf), [pytorch](https://github.com/Sekunde/3D-SIS)]. Predict bbox proposals. training not end-to-end
* [Detection-based] GSPN, CVPR 2019, [[CVPR paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_GSPN_Generative_Shape_Proposal_Network_for_3D_Instance_Segmentation_in_CVPR_2019_paper.pdf), [tensorflow](https://github.com/ericyi/GSPN)]. reconstructs object shapes from shape noisy observations to enforce geometric understanding. training not end-to-end
* [Detection-based] 3D-BoNet, NIPS 2019 [[NIPS paper](https://proceedings.neurips.cc/paper/2019/file/d0aa518d4d3bfc721aa0b8ab4ef32269-Paper.pdf), [tensorflow](https://github.com/Yang7879/3D-BoNet)]. learn a fixed number of bbox, training end-to-end
* [Detection-free] SGPN, CVPR 2018, [[CVPR paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0967.pdf), [tensorflow](https://github.com/laughtervv/SGPN)]. assumes points belonging to the same object instance have similar features. it learns a similarity matrix to predict proposals. but (a) unable to segment adjacent instances of the same class, (b) the all points similarity matrix is expensive, can't process large scene.
* [Detection-free] 3D-MPA, CVPR 2020, [[CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Engelmann_3D-MPA_Multi-Proposal_Aggregation_for_3D_Semantic_Instance_Segmentation_CVPR_2020_paper.pdf)]. learns object proposals from sampled and grouped point features that vote for the same object center, and then consolidates the proposal features using a graph convolutional network enabling higher-level interactions between proposals which result in refined proposal features
* [Detection-free] AS-Net: CVPR 2020, [[CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_End-to-End_3D_Point_Cloud_Instance_Segmentation_Without_Detection_CVPR_2020_paper.pdf)]. uses an assignment module to assign proposal candidates and then eliminates redundant candidates by a suppression network



All other approaches can be called proposal-free approach. Proposal-free approach usually predicts instance-aware features/embeddings (internal features) and uses clustering/grouping to generate instance labels.

### Center Prediction Approach

This approach first predicts a small number of instance centers directly from point cloud input, then generates the final instance bounding box and point masks. 

Pros:

* The computational cost can be lower than proposal-based approach. Proposals or anchors are usually orders of magnitude greater than the real object instances. By predicting object centers first can reduce the computational cost from the beginning.

* A multi-stage approach. We can evaluate the intermediate results (the center heatmap) to check the learning effectiveness

Cons:

* If the point cloud is surface point instead of voxel, the "center" may not be any existing point in the cloud.

Examples:

* GICN: predict center heatmap based on a Gaussian distribution. [[arXiv](https://arxiv.org/abs/2007.09860), [annotated](./refs/2021_GICN.pdf), [code](https://github.com/LiuShihHung/GICN)].  
  * first predicts instance center probability as a heatmap
  * predicts instance size of each center to determine a proper neighborhood for feature extraction (i.e. "size-aware" & "adaptive" feature extraction)
  * estimates the bounding box and point mask for each instance center.

### Clustering approach

Examples:

* PointGroup, CVPR 2020, [[CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.pdf), [annotated](./refs/2020_PointGroup.pdf), [pytorch](https://github.com/dvlab-research/PointGroup)]. On ScanNet v2 and S3DIS datasets.
  * **Semantic**: first uses a semantic segmentation backbone to extract descriptive features and predicts per-point semantic labels
  * **Offset**: uses an offset branch to learn a relative offset to bring points of the same object instance closer to its respective instance centroid, thus enabling better grouping and separation of adjacent objects
  * **Dual-Set Point Grouping**: 
  * ScoreNet to evaluate and select candidate groups. Non-maximum suppression to remove duplicate predictions



## Setup Guides for Different Networks

* [GICN](./gicn-setup.md)
* [PointGroup](./pointgroup-setup.md)