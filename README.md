# T-REX

## Preference Learning

### Installation
Clone the for-LfL branch for https://github.com/dsbrown1331/baselines/tree/for-LfL
inside the learning-rewards-of-learners/learner/ directory and then install baselines

```
cd baselines
pip install -e .
```

Also install gym for the Atari environments:
https://github.com/openai/gym


### Training the reward 

#### Without self attention
```
python LearnAtariNoviceSnippetsSorted.py --env_name breakout --models_dir . --reward_model_path learned_models/breakout-test
```

#### With self attention

```
python LearnAtariNoviceSnippetsSortedSelfAttention.py --env_name breakout --models_dir . --reward_model_path learned_models/self-attention-test
```

### Training PPO on custom learned reward 

#### Without self attention


```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout-self python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/self-attention-test --num_timesteps=2e7 
```

#### With self attention

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout-self python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/self-attention-test --num_timesteps=2e7 --self_attention
```

### Evaluating the learned PPO policy (in progress)

```
python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout-test/checkpoints/3900
```
