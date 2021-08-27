Date: 06/2021

## PointGroup Guide

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

cd lib/pointgroup_ops
python setup.py develop
```

Success!

### Test

Pointgroup repo has command to train on Scannet v2, we need to download partial Scannet. We can find the download script `/torch-point3d/scripts/datasets/download-scannet.py`. And use `python download-scannet.py -o . --type ... --id scene0000_00`. The meaning of each file can be found [here](http://www.scan-net.org/ScanNet/). Then prepare the dataset structure as Pointgroup requires and generate the [scene]_inst_nostuff.pth for training. Remember if we only use one or two scenes in the train dataset, we need to change the batch size to 1 or 2 in `pointgroup/config/pointgroup_run1_scannet.yaml` otherwise the dataloader is empty. Also change the epoch in the yaml file to a small number.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_run1_scannet.yaml 
```



