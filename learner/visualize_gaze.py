import argparse
import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
import atari_head_dataset as ahd 
import utils

if __name__=="__main__":
	env_name = "breakout"
	data_dir = "../data/atari-head/"
	dataset = ahd.AtariHeadDataset(env_name, data_dir)
	
	print(len(dataset.trajectories['breakout'][218]))
	demonstrations, learning_returns, learning_rewards, learning_gaze = utils.get_preprocessed_trajectories(env_name, dataset, data_dir)

	print(len(learning_gaze))
	print(len(demonstrations))
	print(learning_gaze)