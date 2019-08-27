#!/bin/bash
screen -dmS breakout_PPO_500_gaze_coverage_mask_rewards_0.1 bash
screen -S breakout_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=6 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout_500_gaze_coverage_mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout_500_gaze_coverage_mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS hero_PPO_500_gaze_coverage_mask_rewards_0.1 bash
screen -S hero_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=7 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/hero_500_gaze_coverage_mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/hero_500_gaze_coverage_mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS seaquest_PPO_500_gaze_coverage_mask_rewards_0.1 bash
screen -S seaquest_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/seaquest_500_gaze_coverage_mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/seaquest_500_gaze_coverage_mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.1 bash
screen -S spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=1 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/spaceinvaders_500_gaze_coverage_mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders_500_gaze_coverage_mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS mspacman_PPO_500_gaze_coverage_mask_rewards_0.1 bash
screen -S mspacman_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_PPO_500_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/mspacman_500_gaze_coverage_mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/mspacman_500_gaze_coverage_mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS breakout_PPO_500_gaze_coverage_mask_rewards_0.5 bash
screen -S breakout_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=3 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout_500_gaze_coverage_mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout_500_gaze_coverage_mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS hero_PPO_500_gaze_coverage_mask_rewards_0.5 bash
screen -S hero_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=4 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/hero_500_gaze_coverage_mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/hero_500_gaze_coverage_mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS seaquest_PPO_500_gaze_coverage_mask_rewards_0.5 bash
screen -S seaquest_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=5 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/seaquest_500_gaze_coverage_mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/seaquest_500_gaze_coverage_mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.5 bash
screen -S spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=6 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/spaceinvaders_500_gaze_coverage_mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders_500_gaze_coverage_mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS mspacman_PPO_500_gaze_coverage_mask_rewards_0.5 bash
screen -S mspacman_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_PPO_500_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=7 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/mspacman_500_gaze_coverage_mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/mspacman_500_gaze_coverage_mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS breakout_PPO_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S breakout_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout_500_no-gaze_mask_rewards python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout_500_no-gaze__mask_rewards/ --num_timesteps=2e7
"
screen -dmS hero_PPO_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S hero_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=1 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/hero_500_no-gaze_mask_rewards python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/hero_500_no-gaze__mask_rewards/ --num_timesteps=2e7
"
screen -dmS seaquest_PPO_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S seaquest_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/seaquest_500_no-gaze_mask_rewards python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/seaquest_500_no-gaze__mask_rewards/ --num_timesteps=2e7
"
screen -dmS spaceinvaders_PPO_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S spaceinvaders_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=3 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/spaceinvaders_500_no-gaze_mask_rewards python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders_500_no-gaze__mask_rewards/ --num_timesteps=2e7
"
screen -dmS mspacman_PPO_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S mspacman_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_PPO_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=4 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/mspacman_500_no-gaze_mask_rewards python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/mspacman_500_no-gaze__mask_rewards/ --num_timesteps=2e7
"
screen -dmS breakout_PPO_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S breakout_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=5 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout_500_gaze_coverage_no-mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout_500_gaze_coverage_no-mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS hero_PPO_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S hero_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=6 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/hero_500_gaze_coverage_no-mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/hero_500_gaze_coverage_no-mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=7 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/seaquest_500_gaze_coverage_no-mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/seaquest_500_gaze_coverage_no-mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/spaceinvaders_500_gaze_coverage_no-mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders_500_gaze_coverage_no-mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=1 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/mspacman_500_gaze_coverage_no-mask_rewards_0.1 python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/mspacman_500_gaze_coverage_no-mask_rewards_0.1/ --num_timesteps=2e7
"
screen -dmS breakout_PPO_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S breakout_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout_500_gaze_coverage_no-mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout_500_gaze_coverage_no-mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS hero_PPO_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S hero_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=3 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/hero_500_gaze_coverage_no-mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/hero_500_gaze_coverage_no-mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=4 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/seaquest_500_gaze_coverage_no-mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/seaquest_500_gaze_coverage_no-mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=5 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/spaceinvaders_500_gaze_coverage_no-mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders_500_gaze_coverage_no-mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_PPO_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=6 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/mspacman_500_gaze_coverage_no-mask_rewards_0.5 python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/mspacman_500_gaze_coverage_no-mask_rewards_0.5/ --num_timesteps=2e7
"
screen -dmS breakout_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S breakout_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=7 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout_500_no-gaze_no-mask_rewards python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout_500_no-gaze__no-mask_rewards/ --num_timesteps=2e7
"
screen -dmS hero_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S hero_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/hero_500_no-gaze_no-mask_rewards python -m baselines.run --alg=ppo2 --env=HeroNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/hero_500_no-gaze__no-mask_rewards/ --num_timesteps=2e7
"
screen -dmS seaquest_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S seaquest_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=1 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/seaquest_500_no-gaze_no-mask_rewards python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/seaquest_500_no-gaze__no-mask_rewards/ --num_timesteps=2e7
"
screen -dmS spaceinvaders_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S spaceinvaders_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/spaceinvaders_500_no-gaze_no-mask_rewards python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders_500_no-gaze__no-mask_rewards/ --num_timesteps=2e7
"
screen -dmS mspacman_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S mspacman_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_PPO_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=3 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/mspacman_500_no-gaze_no-mask_rewards python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/mspacman_500_no-gaze__no-mask_rewards/ --num_timesteps=2e7
"
