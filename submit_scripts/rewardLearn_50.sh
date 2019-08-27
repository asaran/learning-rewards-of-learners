#!/bin/bash
screen -dmS breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.1 bash
screen -S breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariGazeHumanTrajs.py --env_name breakout --data_dir ../data/atari-head/ --reward_model_path learned_models/breakout_50_gaze_coverage_mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores True
"
screen -dmS hero_rewardLearn_50_gaze_coverage_mask_rewards_0.1 bash
screen -S hero_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariGazeHumanTrajs.py --env_name hero --data_dir ../data/atari-head/ --reward_model_path learned_models/hero_50_gaze_coverage_mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores True
"
screen -dmS seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.1 bash
screen -S seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariGazeHumanTrajs.py --env_name seaquest --data_dir ../data/atari-head/ --reward_model_path learned_models/seaquest_50_gaze_coverage_mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores True
"
screen -dmS spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.1 bash
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariGazeHumanTrajs.py --env_name spaceinvaders --data_dir ../data/atari-head/ --reward_model_path learned_models/spaceinvaders_50_gaze_coverage_mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores True
"
screen -dmS mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.1 bash
screen -S mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariGazeHumanTrajs.py --env_name mspacman --data_dir ../data/atari-head/ --reward_model_path learned_models/mspacman_50_gaze_coverage_mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores True
"
screen -dmS breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.5 bash
screen -S breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariGazeHumanTrajs.py --env_name breakout --data_dir ../data/atari-head/ --reward_model_path learned_models/breakout_50_gaze_coverage_mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS hero_rewardLearn_50_gaze_coverage_mask_rewards_0.5 bash
screen -S hero_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=6 python LearnAtariGazeHumanTrajs.py --env_name hero --data_dir ../data/atari-head/ --reward_model_path learned_models/hero_50_gaze_coverage_mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.5 bash
screen -S seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=7 python LearnAtariGazeHumanTrajs.py --env_name seaquest --data_dir ../data/atari-head/ --reward_model_path learned_models/seaquest_50_gaze_coverage_mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.5 bash
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariGazeHumanTrajs.py --env_name spaceinvaders --data_dir ../data/atari-head/ --reward_model_path learned_models/spaceinvaders_50_gaze_coverage_mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.5 bash
screen -S mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_rewardLearn_50_gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariGazeHumanTrajs.py --env_name mspacman --data_dir ../data/atari-head/ --reward_model_path learned_models/mspacman_50_gaze_coverage_mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS breakout_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 bash
screen -S breakout_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariGazeHumanTrajs.py --env_name breakout --data_dir ../data/atari-head/ --reward_model_path learned_models/breakout_50_no-gaze_mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS hero_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 bash
screen -S hero_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariGazeHumanTrajs.py --env_name hero --data_dir ../data/atari-head/ --reward_model_path learned_models/hero_50_no-gaze_mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS seaquest_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 bash
screen -S seaquest_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariGazeHumanTrajs.py --env_name seaquest --data_dir ../data/atari-head/ --reward_model_path learned_models/seaquest_50_no-gaze_mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS spaceinvaders_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 bash
screen -S spaceinvaders_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariGazeHumanTrajs.py --env_name spaceinvaders --data_dir ../data/atari-head/ --reward_model_path learned_models/spaceinvaders_50_no-gaze_mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS mspacman_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 bash
screen -S mspacman_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_rewardLearn_50_no-gaze_coverage_mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=6 python LearnAtariGazeHumanTrajs.py --env_name mspacman --data_dir ../data/atari-head/ --reward_model_path learned_models/mspacman_50_no-gaze_mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores True
"
screen -dmS breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 bash
screen -S breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=7 python LearnAtariGazeHumanTrajs.py --env_name breakout --data_dir ../data/atari-head/ --reward_model_path learned_models/breakout_50_gaze_coverage_no-mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores False
"
screen -dmS hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 bash
screen -S hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariGazeHumanTrajs.py --env_name hero --data_dir ../data/atari-head/ --reward_model_path learned_models/hero_50_gaze_coverage_no-mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores False
"
screen -dmS seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 bash
screen -S seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariGazeHumanTrajs.py --env_name seaquest --data_dir ../data/atari-head/ --reward_model_path learned_models/seaquest_50_gaze_coverage_no-mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores False
"
screen -dmS spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 bash
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariGazeHumanTrajs.py --env_name spaceinvaders --data_dir ../data/atari-head/ --reward_model_path learned_models/spaceinvaders_50_gaze_coverage_no-mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores False
"
screen -dmS mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 bash
screen -S mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.1 -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariGazeHumanTrajs.py --env_name mspacman --data_dir ../data/atari-head/ --reward_model_path learned_models/mspacman_50_gaze_coverage_no-mask_rewards_0.1 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.1 --metric rewards --mask_scores False
"
screen -dmS breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 bash
screen -S breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariGazeHumanTrajs.py --env_name breakout --data_dir ../data/atari-head/ --reward_model_path learned_models/breakout_50_gaze_coverage_no-mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 bash
screen -S hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariGazeHumanTrajs.py --env_name hero --data_dir ../data/atari-head/ --reward_model_path learned_models/hero_50_gaze_coverage_no-mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 bash
screen -S seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=6 python LearnAtariGazeHumanTrajs.py --env_name seaquest --data_dir ../data/atari-head/ --reward_model_path learned_models/seaquest_50_gaze_coverage_no-mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 bash
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=7 python LearnAtariGazeHumanTrajs.py --env_name spaceinvaders --data_dir ../data/atari-head/ --reward_model_path learned_models/spaceinvaders_50_gaze_coverage_no-mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 bash
screen -S mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_rewardLearn_50_gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=0 python LearnAtariGazeHumanTrajs.py --env_name mspacman --data_dir ../data/atari-head/ --reward_model_path learned_models/mspacman_50_gaze_coverage_no-mask_rewards_0.5 --snippet_len 50 --use_gaze True --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS breakout_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S breakout_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=1 python LearnAtariGazeHumanTrajs.py --env_name breakout --data_dir ../data/atari-head/ --reward_model_path learned_models/breakout_50_no-gaze_no-mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS hero_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S hero_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=2 python LearnAtariGazeHumanTrajs.py --env_name hero --data_dir ../data/atari-head/ --reward_model_path learned_models/hero_50_no-gaze_no-mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS seaquest_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S seaquest_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=3 python LearnAtariGazeHumanTrajs.py --env_name seaquest --data_dir ../data/atari-head/ --reward_model_path learned_models/seaquest_50_no-gaze_no-mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS spaceinvaders_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S spaceinvaders_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=4 python LearnAtariGazeHumanTrajs.py --env_name spaceinvaders --data_dir ../data/atari-head/ --reward_model_path learned_models/spaceinvaders_50_no-gaze_no-mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
screen -dmS mspacman_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S mspacman_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_rewardLearn_50_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "CUDA_VISIBLE_DEVICES=5 python LearnAtariGazeHumanTrajs.py --env_name mspacman --data_dir ../data/atari-head/ --reward_model_path learned_models/mspacman_50_no-gaze_no-mask_rewards --snippet_len 50 --use_gaze False --gaze_loss coverage --gaze_reg 0.5 --metric rewards --mask_scores False
"
