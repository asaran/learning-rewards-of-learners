# T-REX

## Preference Learning

### Training the reward 

#### Without self attention
```
python LearnAtariNoviceSnippetsSorted.py --env_name breakout --models_dir . --reward_model_path learned_models/breakout-test
```

#### With self attention

```
python LearnAtariNoviceSnippetsSortedSelfAttention.py --env_name breakout --models_dir . --reward_model_path learned_models/self-attention-test
```

### Training PPO on custom learned reward (in progress)
```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/breakout-test python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/breakout-test --num_timesteps=2e7
```

### Evaluating the learned PPO policy (in progress)

```
python evaluateLearnedPolicy_condor.py --env_name breakout --checkpoint breakout-test/checkpoints/3900
```
