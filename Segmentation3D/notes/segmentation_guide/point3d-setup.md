Date: 06/2021

# PyTorch Points 3D Guide

[TOC]

This section is a subsection of [3D segmentation section](./segmentation.md).

[PyTorch Points 3D](https://github.com/nicolas-chaulet/torch-points3d) is a high-level framework for many networks on 3D research:

* [blog](https://towardsdatascience.com/torch-points3d-a-unifying-framework-for-deep-learning-on-point-clouds-94115c0be4fb) for quick intro, [blog](https://analyticsindiamag.com/hands-on-guide-to-torch-points3d-a-modular-deep-learning-framework-for-3d-data/) for hands-on guide
* [doc](https://torch-points3d.readthedocs.io/en/latest/)
* [paper](https://arxiv.org/abs/2010.04642), [annotated](./refs/2020_Points3D.pdf)

## Environment Setup

The best practice is to use conda as an environment manager, pip as the package installer, and Poetry as the dependency manager ([source](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry)). Points3D use [Poetry](https://python-poetry.org/) as the package management system. But for simplicity, I just use the old pip style.

On Windows 10 (not succeed yet)

```bash
conda create --prefix H:\envs\points3d python=3.7
conda activate H:\envs\points3d # need to activate in this way
H:\envs\points3d\Scripts\pip.exe install [package] # pip install in this way

git clone https://github.com/nicolas-chaulet/torch-points3d.git # 6/21/2021
cd torch-points3d

conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch # pytorch error with pip, so conda install first
H:\envs\points3d\Scripts\pip.exe install -r requirements.txt # comment out the torch==1.7.0, torch-cluster==1.5.9, torch-geometric==1.7.0, torch-points-kernels==0.7.0, torch-scatter==2.0.6, torch-sparse==0.6.9 lines
H:\envs\points3d\Scripts\pip.exe install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html # there is special compatibility requirements, see https://github.com/rusty1s/pytorch_cluster


```

On Ubuntu:

First follow [link](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130) to make sure cuda 11.1 is installed correctly on Ubuntu 18.04. Since the Ubuntu machine has RTX 3080 GPU, it only supports CUDA Toolkit after 11.0

```bash
conda create --name points3d python=3.7
conda activate points3d

git clone https://github.com/nicolas-chaulet/torch-points3d.git # 6/21/2021
cd torch-points3d

conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.0 -c pytorch
/home/luojiayi/anaconda3/envs/points3d/bin/pip install torch-cluster torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu111.html # these packages should specify PyTorch and CUDA version
/home/luojiayi/anaconda3/envs/points3d/bin/pip install torch-geometric

# comment out the torch, torch-cluster, torch-scatter, torch-sparse, torch-geometric lines in requirements.txt
/home/luojiayi/anaconda3/envs/points3d/bin/pip install -r requirements.txt

# Minkowski engine (https://github.com/NVIDIA/MinkowskiEngine)
conda install openblas-devel -c anaconda
/home/luojiayi/anaconda3/envs/points3d/bin/pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# torchsparse engine (https://github.com/mit-han-lab/torchsparse)
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git

/home/luojiayi/anaconda3/envs/points3d/bin/pip install pycuda # only for registration tasks

python -m unittest -v # unit test
```

All tests passed.