# Maximum Diffusion Reinforcement Learning Repository
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10723320.svg)](https://doi.org/10.5281/zenodo.10723320)

## System Requirements
- Simulations were run on Ubuntu 18.04 and Ubuntu 20.04 operation systems
- Simulations must be run with Python 3 (tested Python 3.6, Python 3.8, and Python 3.9)
- PyTorch can be installed with CUDA or CPU only (tested versions 1.7, 1.11, and 2.0)
- A dockerfile is provided as an alternate to the installation steps below. After installing and running docker desktop on your system, you can instantiate the maxdiff test enviroment with `source run_docker.sh` on Ubuntu or `.\run_docker.bat` on Windows.


## Installation
- MuJoCo must be installed prior to installing python `mujoco_py` package. 
    1. Download free license from MuJoCo website (visit https://www.roboti.us/license.html and click on "Activation Key")
    2. Download the MuJoCo version 2.0 binaries for Linux (https://www.roboti.us/download/mujoco200_linux.zip)
    3. Unzip the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`, and place your license key (the `mjkey.txt` file) at `~/.mujoco/mjkey.txt`.
    4. Add MuJoCo binary path to your environment path `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin`
- Make sure you have the following libaries installed `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
- Python dependencies can be installed with `pip install -r requirements.txt`
    - Note: specified `mujoco_py` version must be used for code to work. Other versions can be modified at your own discretion (e.g. PyTorch with CPU only or with a specific CUDA version)

## Running the code

To train a policy, modify the relevant yaml file for your test scenario (`config/<method>.yaml`). Then, run the following with the desired arguments (possible arguments can be viewed with `python train.py --help`)
```
python3 train.py --env PointMass2D_DoubleIntEnv --method maxdiff --seed 13 --beta 0.1
python3 train.py --env SwimmerEnv_v3 --method maxdiff --seed 13
```

To playback the learned policy, run the following with the arguments matching your training configuration (possible arguments can be viewed with `python train.py --help`)
```
python3 enjoy.py --env PointMass2D_DoubleIntEnv --method maxdiff --mod _H30_alpha5 --beta 0.1
python3 enjoy.py --env SwimmerEnv_v3 --method maxdiff --seed 13 --mod _H40_alpha100
```

Results can also be visualized in the jupyter-notebooks provided in the `notebooks` folder.

## Run time (CPU)

| Environment | Steps | Horizon | Method | Approx. Runtime | 
| -- | -- | --| -- | -- | 
| SwimmerEnv_v3 | 1,000,000 | 40 |  MaxDiff & MPPI | 1 day |
| SwimmerEnv_v3 | 1,000,000 | N/A | SAC | 4 hours |
| HalfCheetahEnv_v3 | 1,000,000 | 10 | MaxDiff & MPPI| 19 hours |
| HalfCheetahEnv_v3 | 1,000,000 | N/A | SAC | 4 hours |
| AntEnv_v3 | 1,000,000 | 20 | MaxDiff & MPPI | 1 day 20 hours |
| AntEnv_v3 |  1,000,000 | N/A | SAC | 10 hours |


## Repository Contents

```
.
├── config/                       # contains environment-specific test setup yaml files
├── data/                         # contains models and data to run notebooks
├── envs/
│   ├── __init__.py
│   ├── build_env.py              # helper function to build environments for training and testing
│   ├── light_swimmer.xml         # specifies alternate physics model (light tail)
│   ├── normalized_actions.py     # wrapper for gym environments to normalize action space
│   ├── pointmass_lib/            # folder contains functions to generate point mass environment and for rendering pointmasss figures
│   ├── replay_buffer.py          # memory buffer for all training all policies
│   └── wrappers.py               # wrappers for gym environments to modify reward functions
├── mpc_lib/                      # model predictive control policy files
│   ├── __init__.py
│   ├── entropy.py                # helper functions for MaxDiff
│   ├── max_diff.py               # MaxDiff policy module        
│   ├── model.py                  # model for MPPI and MaxDiff
│   ├── mppi.py                   # MPPI policy module
│   ├── optimizer.py              # model optimizer for MPPI and MaxDiff
│   └── utils.py                  # helper functions for max_diff and mppi
├── notebooks/                    # jupyter-notebooks for plotting reward curves
├── sac_lib/                      # model-free policy files
│   ├── __init__.py
│   ├── policynetwork.py          # policy network
│   ├── sac_orig.py               # SAC training module (with optimizer)
│   └── sac_networks.py           # SAC training networks
├── Dockerfile                    # Contains setup instructions for docker
├── enjoy.py                      # replay of trained policies (visualization only)
├── eval.py                       # test trained policies (collect data for analysis)
├── LICENSE
├── README.md
├── requirements.txt              # python dependencies
├── run_docker.bat                # run file to start docker container on Ubuntu systems
├── run_docker.sh                 # run file to start docker container on Windows systems
├── train.py                      # main training function
└── utils.py                      # helper functions
```

## Copyright and License

The implementations of MaxDiff contained herein are copyright (C) 2024 - 2025 by Allison Pinosky and Todd Murphey and are distributed under the terms of the GNU General Public License (GPL) version 3 (or later). Please see the LICENSE for more information.

Contact: apinosky@u.northwestern.edu

Lab Info:  
Todd D. Murphey  
https://murpheylab.github.io/  
Northwestern University  
