#!/bin/bash
screen -dmS breakout_eval_500_gaze_coverage_mask_rewards_0.1 bash
screen -S breakout_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout_500_gaze_coverage_mask_rewards_0.1/checkpoints/03900
"
screen -dmS hero_eval_500_gaze_coverage_mask_rewards_0.1 bash
screen -S hero_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name hero --checkpoint hero_500_gaze_coverage_mask_rewards_0.1/checkpoints/03900
"
screen -dmS seaquest_eval_500_gaze_coverage_mask_rewards_0.1 bash
screen -S seaquest_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_500_gaze_coverage_mask_rewards_0.1/checkpoints/03900
"
screen -dmS spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.1 bash
screen -S spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name spaceinvaders --checkpoint spaceinvaders_500_gaze_coverage_mask_rewards_0.1/checkpoints/03900
"
screen -dmS mspacman_eval_500_gaze_coverage_mask_rewards_0.1 bash
screen -S mspacman_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_eval_500_gaze_coverage_mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name mspacman --checkpoint mspacman_500_gaze_coverage_mask_rewards_0.1/checkpoints/03900
"
screen -dmS breakout_eval_500_gaze_coverage_mask_rewards_0.5 bash
screen -S breakout_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout_500_gaze_coverage_mask_rewards_0.5/checkpoints/03900
"
screen -dmS hero_eval_500_gaze_coverage_mask_rewards_0.5 bash
screen -S hero_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name hero --checkpoint hero_500_gaze_coverage_mask_rewards_0.5/checkpoints/03900
"
screen -dmS seaquest_eval_500_gaze_coverage_mask_rewards_0.5 bash
screen -S seaquest_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_500_gaze_coverage_mask_rewards_0.5/checkpoints/03900
"
screen -dmS spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.5 bash
screen -S spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name spaceinvaders --checkpoint spaceinvaders_500_gaze_coverage_mask_rewards_0.5/checkpoints/03900
"
screen -dmS mspacman_eval_500_gaze_coverage_mask_rewards_0.5 bash
screen -S mspacman_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_eval_500_gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name mspacman --checkpoint mspacman_500_gaze_coverage_mask_rewards_0.5/checkpoints/03900
"
screen -dmS breakout_eval_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S breakout_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout_500_no-gaze_mask_rewards/checkpoints/03900
"
screen -dmS hero_eval_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S hero_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name hero --checkpoint hero_500_no-gaze_mask_rewards/checkpoints/03900
"
screen -dmS seaquest_eval_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S seaquest_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_500_no-gaze_mask_rewards/checkpoints/03900
"
screen -dmS spaceinvaders_eval_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S spaceinvaders_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name spaceinvaders --checkpoint spaceinvaders_500_no-gaze_mask_rewards/checkpoints/03900
"
screen -dmS mspacman_eval_500_no-gaze_coverage_mask_rewards_0.5 bash
screen -S mspacman_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_eval_500_no-gaze_coverage_mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name mspacman --checkpoint mspacman_500_no-gaze_mask_rewards/checkpoints/03900
"
screen -dmS breakout_eval_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S breakout_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout_500_gaze_coverage_no-mask_rewards_0.1/checkpoints/03900
"
screen -dmS hero_eval_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S hero_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name hero --checkpoint hero_500_gaze_coverage_no-mask_rewards_0.1/checkpoints/03900
"
screen -dmS seaquest_eval_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S seaquest_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_500_gaze_coverage_no-mask_rewards_0.1/checkpoints/03900
"
screen -dmS spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name spaceinvaders --checkpoint spaceinvaders_500_gaze_coverage_no-mask_rewards_0.1/checkpoints/03900
"
screen -dmS mspacman_eval_500_gaze_coverage_no-mask_rewards_0.1 bash
screen -S mspacman_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_eval_500_gaze_coverage_no-mask_rewards_0.1 -X stuff "python evaluateLearnedPolicy_condor.py --env_name mspacman --checkpoint mspacman_500_gaze_coverage_no-mask_rewards_0.1/checkpoints/03900
"
screen -dmS breakout_eval_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S breakout_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout_500_gaze_coverage_no-mask_rewards_0.5/checkpoints/03900
"
screen -dmS hero_eval_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S hero_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name hero --checkpoint hero_500_gaze_coverage_no-mask_rewards_0.5/checkpoints/03900
"
screen -dmS seaquest_eval_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S seaquest_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_500_gaze_coverage_no-mask_rewards_0.5/checkpoints/03900
"
screen -dmS spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name spaceinvaders --checkpoint spaceinvaders_500_gaze_coverage_no-mask_rewards_0.5/checkpoints/03900
"
screen -dmS mspacman_eval_500_gaze_coverage_no-mask_rewards_0.5 bash
screen -S mspacman_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_eval_500_gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name mspacman --checkpoint mspacman_500_gaze_coverage_no-mask_rewards_0.5/checkpoints/03900
"
screen -dmS breakout_eval_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S breakout_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S breakout_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S breakout_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout_500_no-gaze_no-mask_rewards/checkpoints/03900
"
screen -dmS hero_eval_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S hero_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S hero_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S hero_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name hero --checkpoint hero_500_no-gaze_no-mask_rewards/checkpoints/03900
"
screen -dmS seaquest_eval_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S seaquest_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S seaquest_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S seaquest_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_500_no-gaze_no-mask_rewards/checkpoints/03900
"
screen -dmS spaceinvaders_eval_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S spaceinvaders_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S spaceinvaders_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S spaceinvaders_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name spaceinvaders --checkpoint spaceinvaders_500_no-gaze_no-mask_rewards/checkpoints/03900
"
screen -dmS mspacman_eval_500_no-gaze_coverage_no-mask_rewards_0.5 bash
screen -S mspacman_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "cd /scratch/cluster/asaran/learning-rewards-of-learners/learner/
"
screen -S mspacman_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S mspacman_eval_500_no-gaze_coverage_no-mask_rewards_0.5 -X stuff "python evaluateLearnedPolicy_condor.py --env_name mspacman --checkpoint mspacman_500_no-gaze_no-mask_rewards/checkpoints/03900
"
