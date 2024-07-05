# Evaluation Code on Franka Kitchen Environment
This guidebook outlines the procedures for environment setup, expert demonstrations downloading, and the execution of evaluation scripts.
## Environment Preparation
### Requirements
```bash
# MuJoCo v2.1
# mujoco-py<2.2,>=2.1
# mjrl
# robohive
clip==0.2.0
gdown==4.7.1
gym==0.22.0
moviepy==1.0.3
numpy==1.26.4
omegaconf==2.1.2
pandas==2.2.2
Pillow==9.0.1
scipy==1.11.4
setuptools==68.0.0
tabulate==0.9.0
tqdm==4.66.1
transformers==4.35.2
cython==3.0.0a9
```

### MuJoCo installation

1. Download `MuJoCo v2.1 binaries` from the official [website](https://mujoco.org/)
2. Unzip the downloaded `mujoco210` directory into ~/.mujoco/mujoco210
3. Update bashrc by adding the following lines and source it

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user_name>/.mujoco/mujoco210/bin
export MUJOCO_PY_FORCE_GPU=True
#export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

### mujoco-py installation
Ref: https://github.com/openai/mujoco-py
> [!IMPORTANT]
> mujoco-py does not support versions of MuJoCo after 2.1.0.
```bash
sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
conda install -c anaconda patchelf
pip3 install -U 'mujoco-py<2.2,>=2.1'
```

The `mujoco_py` library is compiled exclusively upon the execution of the "import mujoco_py" command, and this process frequently encounters issues. For resolutions to common problems, please refer to the [Frequent Issues](#Frequent_Issues). Proceed to the next step only after the command "import mujolo_py" completes error-free.

Verifying the correctness of `mujoco-py` installation:
```python
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
# [0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
# 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
sim.step()
print(sim.data.qpos)
#[-1.12164337e-05  7.29847036e-22  1.39975300e+00  9.99999999e-01
#  1.80085466e-21  4.45933954e-05 -2.70LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opendrivelab/.mujoco/mujoco210/bin143345e-20  1.30126513e-19
# -4.63561234e-05 -1.88020744e-20 -2.24492958e-06  4.79357124e-05
# -6.38208396e-04 -1.61130312e-03 -1.37554006e-03  5.54173825e-05
# -2.24492958e-06  4.79357124e-05 -6.38208396e-04 -1.61130312e-03
# -1.37554006e-03 -5.54173825e-05 -5.73572648e-05  7.63833991e-05
# -2.12765194e-05  5.73572648e-05 -7.63833991e-05 -2.12765194e-05]
```

### mjrl installation
```bash
git clone https://github.com/aravindr93/mjrl.git
cd <path_to_mjrl>
```
Replace <path_to_mjrl>/setup/env.yml with the following content:
```bash
name: mjrl-env
channels:
- pytorch
- defaults
dependencies:
- pip
- ipython
- mkl-service
- tabulate
- termcolor
- patchelf
- pip:
  - click
  - cloudpickle
  - gym==0.22
  - ipdb
  - matplotlib
  - mujoco-py<2.2,>=2.1
  - pip
  - pyyaml
  - tqdm
  - wheel
  - scipy
  - transforms3d
```
Build mjrl with pip, the package name built is `mjrl==1.0.0`.
```bash
pip install -e .
```

### robohive installation (contains Franka Kitchen environment)
Ref: https://github.com/vikashplus/robohive/releases/tag/v0.0.5
1. Clone tag v0.0.5 with pre-populated submodule dependencies
```
git clone --branch v0.0.5 --recursive https://github.com/vikashplus/robohive.git
```
2. Modify setup.py:
+ 'gym==0.13' --> 'gym==0.22.0'
+ 'mujoco-py<2.1,>=2.0' --> 'mujoco-py<2.2,>=2.1'
3. Build robohive with pip, and the package name built is `mj_envs`
```bash
pip install -e .
```
### Frequent Issues <a name="frequent_issues"></a>
1. distutils.errors.LinkError: command ‘gcc’ failed with exit status 1

The reason is the absence of the `libgl1-mesa-dev` library.

```bash
sudo aptitude install libgl1-mesa-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
```

2. ImportError: /home/hy/miniconda3/envs/r3m_base/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /lib/x86_64-linux-gnu/libOSMesa.so.8)

```bash
cd ~/miniconda3/envs/<conda_env_path>/lib
rm libstdc++.so.6
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.29 .
ln -s libstdc++.so.6.0.29 libstdc++.so.6
```

3. ImportError: /home/hy/miniconda3/envs/r3m_base/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)
```bash
conda install -c conda-forge libstdcxx-ng=12
```

4. Problems encountered while compiling mujoco_py with Python 3.10:
+ ImportError: /home/hy/miniconda3/envs/RoboEval/lib/python3.10/site-packages/mujoco_py/generated/cymj_2.1.2.14_310_linuxgpuextensionbuilder_310.so: undefined symbol: _PyGen_Send
+ Cython.Compiler.Errors.CompileError: /home/usr/.local/lib/python3.8/site-packages/mujoco_py/cymj.pyx
```bash
pip install cython==3.0.0a9
```

## Downloading Demonstration Data

Downloading the expert demonstration provided by [R3M](https://github.com/facebookresearch/r3m/tree/eval/evaluation). The demonstrations are located [here](https://drive.google.com/drive/folders/108VW5t5JV8uNtkWvfZxEvY2P2QkC_tsf?usp=sharing). The directory tree should look like this: 
```
$<path-to-demonstrations>
└── final_paths_multiview_rb_200
    |—— default  
    |—— left_cap2  
    └── right_cap2
```

## Install and Run Eval Code

If the above was all done correctly, run the following scripts:
```bash
cd mpi_evaluation/franka_kitchen/
pip install -e .
```
Evaluating the success rate of `knob on` task on Franka Kitchen environment with **25** expert demonstration:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=mpi_evaluation/franka_kitchen/MPIEval/core python mpi_evaluation/franka_kitchen/MPIEval/core/hydra_launcher.py hydra/launcher=local hydra/output=local env="kitchen_knob1_on-v3" camera="left_cap2" pixel_based=true embedding=ViT-Small num_demos=25 env_kwargs.load_path=mpi-small bc_kwargs.finetune=false job_name=mpi-small seed=125 proprio=9
```