# T-REX

## Preference Learning

### Installation
Clone the for-LfL branch for https://github.com/asaran/baselines/tree/for-LfL
inside the learning-rewards-of-learners/learner/ directory and then install baselines

```
cd baselines
pip install -e .
```

Also install gym for the Atari environments:
https://github.com/openai/gym

### Demonstration Data
Use checkpointed files for producing the demonstrations. You can download some here: https://github.com/dsbrown1331/learning-rewards-of-learners/releases/tag/atari25 and put them in the learner/models directory. 


### Training the reward 

#### Without self attention
```
python LearnAtariNoviceSnippetsSorted.py --env_name breakout --models_dir . --reward_model_path learned_models/breakout
```

#### With self attention

```
python LearnAtariNoviceSnippetsSortedSelfAttention.py --env_name breakout --models_dir . --reward_model_path learned_models/breakout-SA
```

### Training PPO on custom learned reward 

#### Without self attention


```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout-self python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout --num_timesteps=2e7 
```

#### With self attention

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout-self python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout-SA --num_timesteps=2e7 --self_attention
```

### Evaluating the learned PPO policy 

```
python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout/checkpoints/03900
```
