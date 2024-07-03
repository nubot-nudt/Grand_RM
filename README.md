# Learned Robot Placement

Based on our research paper: **Graph Reinforcement Learning-based Reachability Map Generation for Mobile Manipulation under Flexible Environment**



This code is designed for robotic mobile manipulation. It determines feasible base positions for the robot within the environment, enabling skill transitions. The code implements reinforcement learning, where the input is a graph modeled after the scene and the output includes the base position the robot should reach and signals to start or stop the manipulator. The implementation is based on the Fetch robot in Isaac Sim.



The actor network we trained can output the current position of the robot, while the critic network can evaluate the (state, action) pairs to generate a reachability map.



## Installation

__Requirements:__ 

- [x] isaac-sim **version 2022.1.0**
- [x] The isaac-sim python conda environment `isaac-sim-lrp `;
- [x] pinoccio for inverse kinematics;`conda install pinocchio -c conda-forge`
- [x] You can see requirements.txt for detailed;

### Setup RL algorithm and environments
- For RL, we use the **mushroom-rl** library:
    ```
    conda activate isaac-sim-lrp
    cd mushroom-rl
    pip install -e .
    ```
    
- Finally, install this repository's python package:
    ```
    conda activate isaac-sim-lrp
    cd Grand_RM
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

#### Training:

We provide four parameters, 

```
python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingHyRL num_seeds=1 headless=True
```

- #### task: change task;

  We provide four tasks, from v1 to v4;

  v1: Task with one table, and one goal object and three tabletop obstacles;

  ![Task1_Clip](/home/lu/Desktop/thesis pictures/Task1_Clip.png)

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

  v2: Task with one table, and one goal object, three tabletop obstacles and one ground obstacle like the chair;

  ![Task2_Clip](/home/lu/Desktop/thesis pictures/Task2_Clip.png)

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v21 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

  v3: Semi-enclosed task, the goal object was surrounded by three obstacle plank;

  ![Task3_Clip](/home/lu/Desktop/thesis pictures/Task3_Clip.png)

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v31 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

  v4：Enclosed task, the goal object was surrounded by three obstacle plank and a lid;

  ![Task4_Clip](/home/lu/Desktop/thesis pictures/Task4_Clip.png)

  ```
  python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v41 train=FetchMultiObjFetchingGAT num_seeds=1 headless=True
  ```

  reach task: The robot need to reach to goal;

  ![reach_task](/home/lu/Desktop/thesis pictures/reach_task.png)

  ```
  python learned_robot_placement/scripts/train__reach.py task=FetchReaching train=FetchReachingBHyRL num_seeds=1 headless=True
  ```

  

- #### train: change baseline methods;

  We provide five baselines which are 

  1. BHyRL:

     ```
     python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingBHyRL num_seeds=1 headless=True
     ```

  2. HyRL:

     ```
     python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingHyRL num_seeds=1 headless=True
     ```

  3. AWAC:

     ```
     python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingAWAC num_seeds=1 headless=True
     ```

  4. SAC-Hybrid:

     ```
     python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingSAC_hybrid num_seeds=1 headless=True
     ```

  5. SAC:

     ```
     python learned_robot_placement/scripts/train_task_fetch.py task=FetchMultiObjFetching_v11 train=FetchMultiObjFetchingSAC num_seeds=1 headless=True
     ```

- #### num_seeds: change num_seeds;

  You can set differenet num_seeds;

- #### headless: change mode;

  You can set headless to False if you want to visualize the training process, or True otherwise which will be sightly faster. 

#### Testing:

- You can test the model, with command: Notice that the model should correspond to the task, or the success will fall;

  ```
  python learned_robot_placement/scripts/train_task.py task=FetchMultiObjFetching_v15 train=FetchMultiObjFetchingGAT test=True checkpoint=/v1/BHyRL/v15/2024-06-08-02-24-32/agent-520.msh
  ```

  

### Possible problems：
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
It might be caused by the version of isaac-sim；
```

### Configuration and command line arguments

- We use [Hydra](https://hydra.cc/docs/intro/) to manage the experiment configuration
- Common arguments for the training scripts are: `task=TASK` (Selects which task to use) and `train=TRAIN` (Selects which training config to use).
- You can check current configurations in the `/cfg` folder

For more details about the code structure, have a look at the OmniIsaacGymEnvs docs: https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/tree/main/docs

## Add-ons:

To generate sampled reachability and base-placement maps for a mobile manipulator (as visualized in the paper), have a look at: https://github.com/iROSA-lab/sampled_reachability_maps



## Troubleshooting

- **"[Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setGlobalPose: pose is not valid."** This error can be **ignored** for now. Isaac-sim 2022.1.0 has some trouble handling the set_world_pose() function for RigidPrims, but this doesn't affect the experiments.


(This fork contains the implementation of **Robot Learning of Mobile Manipulation With Reachability Behavior Priors** https://github.com/iROSA-lab/rlmmbp)
