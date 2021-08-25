Date: 06/2021

# PyTorch Points 3D Guide

[TOC]

This section is a subsection of [3D segmentation section](./segmentation.md).

[PyTorch Points 3D](https://github.com/nicolas-chaulet/torch-points3d) is a high-level framework for many networks on 3D research:

* [doc](https://torch-points3d.readthedocs.io/en/latest/)
* [paper](https://arxiv.org/abs/2010.04642)
* [video](https://www.youtube.com/watch?v=qKGyykBE6oU)
* [blog](https://towardsdatascience.com/torch-points3d-a-unifying-framework-for-deep-learning-on-point-clouds-94115c0be4fb) for quick intro, [blog](https://analyticsindiamag.com/hands-on-guide-to-torch-points3d-a-modular-deep-learning-framework-for-3d-data/) for hands-on guide

## Github Setup

* We want to fork points3D and get up to date with the original repo
* We want to add the torch-points3d repo as a subdirectory in my big 3D/Segmentation3D repo
  * Fork the repo on github web, now this forked repo has a link `https://github.com/symphonylyh/torch-points3d.git`
  * Go to the big 3D repo, add the forked repo as a subtree: `git subtree add --prefix Segmentation3D/Points3D https://github.com/symphonylyh/torch-points3d.git master --squash`. This will create a folder `3D/Segmentation3D/Points3D` which is the master branch of the torch-points3d repo. use `--squash` to merge as just one commit.
  * Commit it in Github Desktop or by `git push origin master`. Now this repo is added as a subrepo
  * say the local repo is A, the forked repo is B, the original repo is C. ABC are different repos.
  * General push just goes to A; to update A <--> B, `git subtree push --prefix Segmentation3D/Points3D https://github.com/symphonylyh/torch-points3d.git master` and `git subtree pull --prefix Segmentation3D/Points3D https://github.com/symphonylyh/torch-points3d.git master --squash`; to update B <-- C, on github web, click "Fetch Upstream" --> "Fetch and merge"
* A subdirectory can also be extracted to be a separate repo by subtree. Check out that later.

## Environment Setup

The best practice is to use conda as an environment manager, pip as the package installer, and Poetry as the dependency manager ([source](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry)). Points3D use [Poetry](https://python-poetry.org/) as the package management system. But for simplicity, I just use the old pip style.

### Windows 10 (no sparse conv backend)

see [Github issue](https://github.com/nicolas-chaulet/torch-points3d/issues/566) here for the correct instructions

```bash
conda create --prefix H:\envs\points3d python=3.7
conda activate H:\envs\points3d # need to activate in this way
H:\envs\points3d\Scripts\pip.exe install [package] # pip install in this way

git clone https://github.com/nicolas-chaulet/torch-points3d.git # 6/21/2021
cd torch-points3d

conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch # pytorch error with pip, so conda install first
H:\envs\points3d\Scripts\pip.exe install -r requirements.txt # comment out the torch==1.7.0, torch-cluster==1.5.9, torch-geometric==1.7.0, torch-points-kernels==0.7.0, torch-scatter==2.0.6, torch-sparse==0.6.9 lines
```

When installing `torch-sparse`, there is error message `H:/envs/points3d/lib/site-packages/torch/include\torch/csrc/jit/ir/ir.h(1347): error: member "torch::jit::ProfileOptionalOp::Kind" may not be initialized`. Recall from the [GICN notes](./gicn-setup.md), a solution was found to be just [comment out the line](https://github.com/facebookresearch/detectron2/issues/9#issuecomment-735284929). 

After fix this, new error message `H:\envs\points3d\lib\site-packages\torch\include\pybind11\cast.h(1449): error: expression must be a pointer to a complete object type`. From this [issue](https://github.com/pytorch/pytorch/issues/11004#issuecomment-717780050), we need to change the line from `explicit operator type&() { return *(this->value); }` to `explicit operator type&() { *((type *)(this->value)); }`

After fixing these two files in the virtual environments, `torch-sparse` should be good. Previously I still failed but it was found due to my CUDA version. I installed v10.0 (so nvcc is 10.0) but specifying v10.2 everywhere. After install the v10.2 CUDA toolkit everything is fine. Run the following commands:

```bash
# packages that need specify pytorch and cuda version when pip (torch-sparse takes longer)
H:\envs\points3d\Scripts\pip.exe install torch-cluster torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html # there is special compatibility requirements, see https://github.com/rusty1s/pytorch_cluster

H:\envs\points3d\Scripts\pip.exe install torch-geometric torch-points-kernels
H:\envs\points3d\Scripts\pip.exe install -r requirements.txt

H:\envs\points3d\Scripts\pip.exe install pycuda

python -m unittest -v # tests related to sparseconv will fail, but others should mostly be ok

# try train a pointnet, this can also verify the basic functionalities
# note that windows doesn't support multi-processing in data loader, so change the /conf/training/default.yaml and default_reg.yaml, num_workers=0, batch_size=2. For all other models and datasets, change these too on Windows
python train.py task=segmentation models=segmentation/pointnet2 model_name=pointnet2_charlesssg data=segmentation/shapenet-fixed

# test pointgroup
python train.py task=panoptic models=panoptic/pointgroup model_name=PointGroup data=panoptic/s3disfused
```

At the time being, both sparse conv backend options (Minkowski Engine, or torchsparse) are not supported on Windows. So on Windows we can't use `from torch_points3d.applications.sparseconv3d import SparseConv3d` yet.

**Note that windows doesn't support multi-processing in data loader, so change the /conf/training/default.yaml and default_reg.yaml, num_workers=0, batch_size=2. For all other models and datasets, change these too on Windows**

For visualization in Jupyter Lab,

```bash
conda install -c conda-forge jupyterlab

# Jupyer Lab before version 3.0
conda install nodejs # Node.js
jupyter labextension install @pyviz/jupyterlab_pyviz # pyviz extension

# Jupyer Lab after version 3.0
conda install -c pyviz pyviz_comms 

conda install -c conda-forge pyvista
conda install -c pyviz panel

jupyter lab # to start in browser and open notebooks
```

Note: on Windows there may be a h5py DLL error, try uninstall it from pip and install from conda.

### Ubuntu

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

python train.py task=segmentation models=segmentation/pointnet2 model_name=pointnet2_charlesssg data=segmentation/shapenet-fixed # dummy training
```

All tests passed.

For visualization in Jupyter Lab,

```bash
conda install -c conda-forge jupyterlab pyvista
conda install -c pyviz pyviz_comms  panel

jupyter lab # to start in browser and open notebooks
```

## Usage

Project structure:

```bash
├─ benchmark               # Output from various benchmark runs
├─ conf                    # All configurations for training nad evaluation leave there
├─ notebooks               # A collection of notebooks that allow result exploration and network debugging
├─ docker                  # Docker image that can be used for inference or training
├─ docs                    # All the doc
├─ eval.py                 # Eval script
├─ find_neighbour_dist.py  # Script to find optimal #neighbours within neighbour search operations
├─ forward_scripts         # Script that runs a forward pass on possibly non annotated data
├─ outputs                 # All outputs from your runs sorted by date
├─ scripts                 # Some scripts to help manage the project
├─ torch_points3d
	├─ applications        # 
    ├─ core                # Core components
    ├─ datasets            # All code related to datasets
    ├─ metrics             # All metrics and trackers
    ├─ models              # All models
    ├─ modules             # Basic modules that can be used in a modular way
    ├─ utils               # Various utils
    └─ visualization       # Visualization
├─ test
└─ train.py                # Main script to launch a training
```

Some python basics that I wasn't very familiar about:

**Multiple inheritance**

`class C(A, B): pass` means attributes & methods inheritance from both A and B classes. Note that the method resolution order is bottom-up, left-right, i.e. C --> A --> B.

use `super()` to call base class's method, e.g. `super().__init__()`. If have multiple bases, the first base is referred by `super()`. Or explicitly call by base class name, `BaseClass.__init__(self)`.

**Abstract class**

Across the code we may often see `from abc import ABC, abstractmethod`. ABC is a built-in python module for Abstract Base Class, which allows to define abstract class (can't be instantiated and can require pure virtual methods in subclasses) like C++.

```python
from abc import ABC, abstractmethod
 
class AbstractClass(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()
    
    @abstractmethod # with this decorator, any derived class MUST implement this to be valid 
    def do_something(self):
        print("Base implementation!")
        pass
    
class Derived(AbstractClass):
    def do_something(self):
        super().do_something() # derived class can call ABC's abstract method
        print("Derived implementation!")
        return self.value+1
```

**@property decorator**

```python
class Student:
    def __init__(self, name):
        self._name=name

    @property # use @property decorator for the getter function
    def name(self):
        return self._name

    @name.setter   # use [property-name].setter for the setting function
    def name(self, value):
        self._name = value
s = Student("Name")
s.name
s.name = "He"
```

**filter()**

filter(condition, sequence), condition is a function or lambda function that will return true/false, sequence is the list before filtering. Example

```python
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
```

**repr**

```python
def __repr__(self): # this is to override what's printed when print(ClassInstance)
    return self.__class__.__name__ # this is used to print the class name
```

### YAML Config

Points3D use YAML files for hierarchical configuration of models, datasets, etc. It uses OmegaConf library (actually it follows Facebook Hydra and that's why). See a concrete Hydra example for ML [here](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b).

For example, 

```python
# conf/config.yaml
db:
  driver: mysql
  user: omry
  pass: secret
  test: ${dataset.name} # use ${} to refer anothe variable, where `dataset` is the @package name, `name` is the key name

# app.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config") # use this syntax sugar to specify config file path and Hydra will load the 'config.yaml' file when run this function
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg)) # this will print the YAML file
    # access by:
    cfg.db.driver # object attribute
    cfg['db']['driver'] # dict

# or instead, you can just load the YAML file to access the configs
opts = Omegaconf.load("conf/config.yaml")

if __name__ == "__main__":
    my_app()
    
# command line
python app.py db.user=root db.pass=1234 # can be overwrite by command line
```

Hierarchical example:

```python
# file structure
├── conf
│   ├── config.yaml
│   ├── db
│   │   ├── mysql.yaml
│   │   └── postgresql.yaml
│   └── __init__.py
└── my_app.py

# conf/config.yaml
defaults:
  - db: mysql # use `???` to make this field a mandatory from command line
  - db/mysql # this means non-overridable defaults
  - /db/mysql # this will look up a directory level
  - [folder_name]: [file_name]
...

# multirun
python app.py --multirun db=mysql,postgresql # easily by this line it can run the program with different configs
```

when load config.yaml, the keyword 'defaults' tell it to load the configs in `/db/mysql.yaml`, key is folder name, value is file name. This is called config groups. To define these config groups you need to include a special directive at the beginning of every file `# @package [folder name]`.

For example, line `- task: segmentation` will search for `/task/segmentation.yaml` to fill in the configurations.

Thus by changing this 'defaults' in YAML or in command line can switch between different configs.

### Train

The main entrance is `train.py`, which loads and overrides the `conf/config.yaml` and calls the real trainer `torch_points3d/trainer.py`.

Windows has dataloader issue with multi-processing, so I changed `trainer.py:108` to be special on Windows. Search `[HHH]` for edits.