# from Flamingo - jaykorea

https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot

changed names from "flamingo" to "amazing"

## Setup
- This repo is tested on Ubuntu 20.04, and I recommend you to install 'local install'
### 1. Install Isaac Sim
  ```
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html
  ```
### 2. Install Isaac Lab
  ```
  https://github.com/isaac-sim/IsaacLab
  ```

### 3. Install lab.flamingo package
i. clone repository
   ```
   git clone https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot
   ```
ii. install lab.flamingo pip package by running below command
   - run it on 'lab.flamingo' root path
   ```
   conda activate env_isaaclab # change to you conda env
   pip install -e .
   ```
iii. Unzip assets(usd asset) on folder
   - Since git does not correctly upload '.usd' file, you should manually unzip the usd files on assests folder
   ```
    path example: lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev01_4_1/
   ```

## Launch script
### Train flamingo
  - run it on 'lab.flamingo' root path
  ```
    python scripts/co_rl/train.py --task {task name} --algo ppo --num_envs 4096 --headless --num_policy_stacks {stack number on policy obs} --num_critic_stacks {stack number on critic obs}
  ```
### Train example - track velocity
  ```
    python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v1-ppo --algo ppo --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
  ```
### play flamingo
  - run it on 'lab.flamingo' root path
  ```
    python scripts/co_rl/play.py --task {task name} --algo ppo --num_envs 64 --num_policy_stacks {stack number on policy obs} --num_critic_stacks {stack number on critic obs} --load_run {folder name} --plot False
  ```
### play example - track velocity
  ```
    python scripts/co_rl/play.py --task Isaac-Velocity-Flat-Flamingo-Play-v1-ppo --algo ppo --num_envs 64 --num_policy_stacks {stack number on policy obs} --num_critic_stacks {stack number on critic obs} --load_run 2025-03-16_17-09-35 --plot False
  ```