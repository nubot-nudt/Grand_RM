# Graph Reinforcement Learning-based Reachability Map Generation for Mobile Manipulation under Flexible Environment

Source code for our research paper: Graph Reinforcement Learning-based Reachability Map Generation for Mobile Manipulation under Flexible Environment



This code is designed for mobile manipulation. It determines feasible base positions for the robot within the environment, enabling skill transitions. The actor network we trained can output the current position of the robot, while the critic network can evaluate the (state, action) pairs to generate a reachability map.



The code implements reinforcement learning, where the input is a graph modeled after the scene and the output includes the base position the robot should reach and signals to start or stop the manipulator. The implementation is based on the Fetch robot in Isaac Sim.

## Installation

__Isaac sim simulation environment:__ 

- [x] Isaac-sim version 2022.1.0;
- [x] Install isaac-sim python conda environment,  floolw the instruction https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda;
- [x] Install pinocchio `conda install pinocchio -c conda-forge`；

### Setup RL algorithm and environments
- For RL, we use the **mushroom-rl** library [4]. To install mushroom-rl:
    ```
    conda activate isaac-sim-lrp
    cd mushroom-rl
    pip install -e .
    ```
    
- Finally, install this repository's python package:
    ```
    cd learned_robot_placement
    pip install -e .
    ```

## Experiments

### Launching the experiments
- Activate the conda environment:
    ```
    conda activate isaac-sim-lrp
    ```
    
- source the isaac-sim conda_setup file:
    ```
    source ~/.local/share/ov/pkg/isaac_sim-2022.1.0/setup_conda_env.sh
    ```
    
- To test the installation, an example random policy can be run:
    ```
    python learned_robot_placement/scripts/random_policy.py
    ```
    
- To train a Grand-RM model:

    ```
    python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
    ```

- To train a baseline model: change parameters train, we provide five baseline algorithms;

  ```
  # BHyRL
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingBHyRL num_seeds=1 headless=True
  
  # SAC_hybrid
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingSAC_hybrid num_seeds=1 headless=True
  
  # AWAC
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingAWAC num_seeds=1 headless=True
  
  # SAC
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingSAC num_seeds=1 headless=True
  
  # HyRL
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingHyRL num_seeds=1 headless=True
  ```

- Train other tasks:

  We have four kind of tasks:

  v1 is: Task with one table, one goal object and there tebletop obstacles, the robot needs to pick up the object with red circle;

  You just need to set task=FetchMultiObjFetching_v11;

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

  ![Task1_Clip](/home/lu/Music/grand_rm/videos/Task1_Clip.png)

  v2 is: Task with one table, a ground obstacle the chair, one goal object and three tabletop obstacles;

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v21 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

  ![Task2_Clip](/home/lu/Desktop/thesis pictures/Task2_Clip.png)

  v3 is: Semi-enclosed task, the goal object is surrounded by three obstacle plank;

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v31 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

  ![Task3_Clip](/home/lu/Desktop/thesis pictures/Task3_Clip.png)

  v4 is Enclosed task, the goal object is surrounded by three obstacle plank and a lid;

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v41 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

![Task4_Clip](/home/lu/Desktop/thesis pictures/Task4_Clip.png)

- If visualization of the results is needed, headless should be set to True, the training process will be faster if the headless if False:
    
    ```
    python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v41 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
    ```
    
- To train reach task:
    
    ![reach_task](/home/lu/Desktop/thesis pictures/reach_task.png)
    
    ```
    python learned_robot_placement/scripts/train_task_reach.py task=FetchReaching train=FetchReachingBHyRL num_seeds=1 headless=False
    ```

### Possible prblems：
```commandline
Error executing job with overrides: []
Traceback (most recent call last):
  File "learned_robot_placement/scripts/random_policy.py", line 50, in parse_hydra_configs
    env = IsaacEnvMushroom(headless=headless,render=render,sim_app_cfg_path=sim_app_cfg_path)
  File "/home/lu/Desktop/embodied_ai/rlmmbp/learned_robot_placement/envs/isaac_env_mushroom.py", line 45, in __init__
    "height": RENDER_HEIGHT})
  File "/home/lu/.local/share/ov/pkg/isaac_sim-2022.2.0/exts/omni.isaac.kit/omni/isaac/kit/simulation_app.py", line 191, in __init__
    create_new_stage()
  File "/home/lu/.local/share/ov/pkg/isaac_sim-2022.2.0/exts/omni.isaac.kit/omni/isaac/kit/utils.py", line 65, in create_new_stage
    return omni.usd.get_context().new_stage()
AttributeError: 'NoneType' object has no attribute 'new_stage'
可能出现问题的原因是Issac-sim的版本；
```

(This fork contains the implementation of **Robot Learning of Mobile Manipulation With Reachability Behavior Priors** (BHyRL)** [1] https://github.com/iROSA-lab/rlmmbp/tree/main)



