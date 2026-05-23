CUDA_VISIBLE_DEVICES=1 python scripts/co_rl/play.py --task Isaac-TrackJUMP-Flat-Amazing-v0-ppo --algo ppo --num_envs 64 --num_policy_stacks 10 --num_critic_stacks 10 --load_run 2026-05-20_22-52-30 --plot False --video --headless

CUDA_VISIBLE_DEVICES=1 python scripts/co_rl/train.py --task Isaac-TrackJUMP-Flat-Amazing-v0-ppo --algo ppo --num_envs 25000 --num_policy_stacks 10 --num_critic_stacks 10 --headless

CUDA_VISIBLE_DEVICES=1 python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Amazing-v0-ppo --algo ppo --num_envs 25000 --num_policy_stacks 10 --num_critic_stacks 10 --headless

 CUDA_VISIBLE_DEVICES=1 python scripts/co_rl/play.py --task Isaac-Velocity-Flat-Amazing-v0-ppo --algo ppo --num_envs 64 --num_policy_stacks 10 --num_critic_stacks 10 --load_run 2026-05-20_17-52-48 --plot False --video --headless