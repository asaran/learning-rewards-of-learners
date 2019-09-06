import argparse
# import agc.dataset as ds
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
# import agc_demos
# from utils import get_atari_head_demos
import atari_head_dataset as ahd 
import utils
import LearnAtariNoviceSnippetsSorted as novice 
from cnn import Net
#cheat and sort them to see if it helps learning
#sorted_demos = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
#sorted_returns = sorted(learning_returns)
#print(sorted_returns)
#plt.plot(sorted_returns)


# Create training data by taking random 50 length crops of trajectories, computing the true returns and adding them to the training data with the correct label.
# def create_training_data(demonstrations, returns, gaze_maps, n_train):
def create_training_data(demonstrations, returns, rewards, gaze_coords, n_train, use_gaze, snippet_length, comp_metric):
	training_obs = []
	training_labels = []
	training_gaze = []
	num_demos = len(demonstrations)
	for n in range(n_train):
		ti, tj = 0, 0
		r_i, r_j = 0, 0   
		rew_i, rew_j = 0, 0
		
		if comp_metric=='returns':
			x_i, x_j = r_i, r_j
		elif comp_metric=='rewards':
			x_i, x_j = rew_i, rew_j
		else:
			print('invalid trajectory comparison metric!')
			exit(1)
		
		#only add trajectories that are different returns
		# while(ti == tj):
		# while(r_i == r_j):
		while(x_i == x_j):
			#pick two random demonstrations
			ti = np.random.randint(num_demos)
			tj = np.random.randint(num_demos)
			#print(ti, tj)
			#create random snippets
			# print('# of demos: ', len(demonstrations))
			# print('len demo: ',len(demonstrations[ti]), len(demonstrations[tj]))

			# ensure that sampled demo length is greater than the requested snippet length
			if(len(demonstrations[ti])<snippet_length or len(demonstrations[tj])<snippet_length):
				continue

			ti_start = np.random.randint(len(demonstrations[ti])-snippet_length)
			tj_start = np.random.randint(len(demonstrations[tj])-snippet_length)
			#print("start", ti_start, tj_start)

			traj_i = demonstrations[ti][ti_start:ti_start+snippet_length]
			traj_j = demonstrations[tj][tj_start:tj_start+snippet_length]

			# print(returns[ti][ti_start:ti_start+snippet_length])
			# print((returns[ti]))
			r_i = returns[ti]
			r_j = returns[tj]

			rew_i = sum(rewards[ti][ti_start:ti_start+snippet_length])
			rew_j = sum(rewards[tj][tj_start:tj_start+snippet_length])

			if use_gaze:
				gaze_i = gaze_coords[ti][ti_start:ti_start+snippet_length]
				gaze_j = gaze_coords[tj][tj_start:tj_start+snippet_length]

			if comp_metric=='returns':
				x_i, x_j = r_i, r_j
			elif comp_metric=='rewards':
				x_i, x_j = rew_i, rew_j


		# labels will be created differently
		# if r_i > r_j:
		if x_i > x_j:
			label = 0
		else:
			label = 1
		#print(label)
		#TODO: maybe add indifferent label?
		training_obs.append((traj_i, traj_j))
		training_labels.append(label)
		if use_gaze:
			training_gaze.append((gaze_i, gaze_j))

	return training_obs, training_labels, training_gaze




# In[111]:

def gaze_loss_KL(true_gaze, conv_gaze):
	true_gaze = torch.from_numpy(true_gaze).float().to(device)
	conv_gaze = torch.from_numpy(conv_gaze).float().to(device)
	loss = F.kl_div(true_gaze, conv_gaze)
	return loss

# wasserstein loss or Earth mover's distance
def gaze_loss_EMD(true_gaze, conv_gaze):
	# both maps are size snippet_len,batch_size,7,7
	# iterate over the snippet length and add gaze loss per map comparison
	from pyemd import emd, emd_samples
	from scipy.stats import wasserstein_distance

	loss = 0
	for t,c in zip(true_gaze,conv_gaze):
		for tb, cb in zip(t,c):
			
			# flatten input maps
			maps = [img.ravel() for img in [tb, cb]]

			# compute EMD using values
			if np.isnan(true_gaze).any():
				print('True Gaze is nan')
				print(true_gaze)
			if np.isnan(conv_gaze).any():
				print('Conv Gaze is nan')
			d1 = emd_samples(maps[0], maps[1]) # 25.57794401220945
			d2 = wasserstein_distance(maps[0], maps[1]) # 25.76187896728515

			loss += d1

			# compute EMD using distributions
			# N_BINS = 256
			# hists = [np.histogram(img, N_BINS, density=True)[0].astype(np.float64) for img in maps]

			# mgrid = np.meshgrid(np.arange(N_BINS), np.arange(N_BINS))
			# metric = np.abs(mgrid[0] - mgrid[1]).astype(np.float64)

			# emd(hists[0], hists[1], metric) # 25.862491463680065

	# loss = d1
	return loss

# coverage based loss, only penalize if conv_gaze is not a superset of true_gaze
def gaze_loss_coverage(true_gaze, conv_gaze):
	loss = 0

	for t,c in zip(true_gaze,conv_gaze):
		for tb, cb in zip(t,c):
			# flatten input maps
			maps = [img.ravel() for img in [tb, cb]]

			# iterate over all coordinates of the true_gaze map
			for i in range(len(maps[0])):
				# compare pixel values between true and conv gaze map
				# add penalty if difference between true_gaze and conv_gaze pixel values is greater than a threshold (0.5)
				# ignore true gaze map pixels that have a zero value (no attention)
				if maps[0][i]>0 and abs(maps[0][i]-maps[1][i])>0.5:
					loss+=abs(maps[0][i]-maps[1][i])

			# normalize loss by batch size? Currently batch size is zero.

	return loss

# Now we train the network. I'm just going to do it one by one for now. Could adapt it for minibatches to get better gradients
def learn_reward(reward_network, optimizer, training_data, num_iter, l1_reg, checkpoint_dir, gaze_loss_type, gaze_reg, gaze_dropout):
	training_inputs, training_outputs, training_gaze26 = training_data

	# generate all gaze heatmaps here
	# if gaze_dropout:
	# 	training_gaze26 = utils.get_all_gaze_heatmaps(training_gaze, 26)
		# training_gaze11 = utils.get_gaze_heatmap(training_gaze, 11)

	# if gaze_loss_type is not None:
	# 	training_gaze7 = utils.get_all_gaze_heatmaps(training_gaze, 7)

	# multiplier for gaze loss
	# gaze_reg = 0.5
	if gaze_loss_type=='EMD':
		gaze_loss = gaze_loss_EMD
	elif gaze_loss_type=='coverage':
		gaze_loss = gaze_loss_coverage
	elif gaze_loss_type=='KL':
		gaze_loss = gaze_loss_KL
	# else:
	# 	print('Invalid gaze loss type')
	# 	exit(1)

	#check if gpu available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# Assume that we are on a CUDA machine, then this should print a CUDA device:
	print(device)
	loss_criterion = nn.CrossEntropyLoss()
	#print(training_data[0])
	cum_loss = 0.0
	training_data = list(zip(training_inputs, training_outputs))
	print('training data: ', type(training_inputs[0]))
	for epoch in range(num_iter):
		np.random.shuffle(training_data)
		training_obs, training_labels = zip(*training_data)

		for i in range(len(training_labels)):
			traj_i, traj_j = training_obs[i]
			if gaze_dropout:
				# gaze_coord pairs are in training_gaze[i] for each trajectory
				# gaze_coords_i, gaze_coords_j = training_gaze[i]

				gaze26_i, gaze26_j = training_gaze26[i]
				# gaze11_i, gaze11_j = training_gaze11[i]

			labels = np.array([[training_labels[i]]])
			
			# print('length of traj and gaze:', len(traj_i), len(gaze26_i))

			# print(traj_i)
			traj_i = np.array(traj_i)
			traj_j = np.array(traj_j)
			traj_i = torch.from_numpy(traj_i).float().to(device)
			traj_j = torch.from_numpy(traj_j).float().to(device)
			labels = torch.from_numpy(labels).to(device)
			
			#zero out gradient
			optimizer.zero_grad()

			#forward + backward + optimize
			if gaze_dropout:
				# print('forward pass')
				outputs, abs_rewards, conv_map_i, conv_map_j = reward_network.forward(traj_i, traj_j, gaze26_i, gaze26_j, train=True)
			else:
				outputs, abs_rewards, _, _ = reward_network.forward(traj_i, traj_j, train=True)			
			outputs = outputs.unsqueeze(0)

			# print('computing loss')
			output_loss = loss_criterion(outputs, labels)
			# print('output loss: ', output_loss)

			# if i<len(training_gaze7):
			if gaze_loss_type is None:
				loss = output_loss + l1_reg * abs_rewards
			
			else:
				# ground truth human gaze maps (7x7)
				gaze7_i, gaze7_j = training_gaze26[i]
				# gaze26_i, gaze26_j = training_gaze26[i]
				# gaze11_i, gaze11_j = training_gaze11[i]
				
				# get gaze heatmaps of size 7x7 for a trajectory
				# gaze7_i = utils.get_gaze_heatmap(gaze_coords_i, 7)
				# gaze7_j = utils.get_gaze_heatmap(gaze_coords_j, 7)

				gaze7_i = np.array(gaze7_i)
				gaze7_j = np.array(gaze7_j)

				# get normalized conv map output (7x7)
				# gaze_map_i = reward_network.conv_map(traj_i, gaze7_i, train=True).cpu().detach().numpy()
				# gaze_map_j = reward_network.conv_map(traj_j, gaze7_j, train=True).cpu().detach().numpy()
				# print('conv map shape: ', gaze_map_i.shape) #(50,1,7,7)
				
				gaze_loss_i = gaze_loss(gaze7_i, conv_map_i)
				gaze_loss_j = gaze_loss(gaze7_j, conv_map_j)

				gaze_loss_total = (gaze_loss_i + gaze_loss_j)
				# gaze_loss_total = np.array([[gaze_loss_total]])
				# gaze_loss_total = torch.from_numpy(gaze_i).float().to(device)
				gaze_loss_total = torch.tensor(gaze_loss_total)
				print('gaze loss: ', gaze_loss_total.data)            

				loss = output_loss + l1_reg * abs_rewards + gaze_reg * gaze_loss_total

			# print('total loss: ', loss.cpu().detach().numpy())  

			# else:
			# 	loss = output_loss + l1_reg * abs_rewards
			# 	print('total loss: ', loss)  

			loss.backward()
			optimizer.step()

			#print stats to see if learning
			item_loss = loss.item()
			cum_loss += item_loss
			if i % 500 == 499:
				#print(i)
				print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
				#print(abs_rewards)
				cum_loss = 0.0
				print("check pointing")
				torch.save(reward_net.state_dict(), checkpoint_dir+"/"+str(i)+'.pth')
	print("finished training")



def calc_accuracy(reward_network, training_inputs, training_outputs, training_gaze26, gaze_dropout):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# loss_criterion = nn.CrossEntropyLoss()
	#print(training_data[0])
	num_correct = 0.

	# if gaze_dropout:
	# 	training_gaze26 = utils.get_all_gaze_heatmaps(training_gaze, 26)
		# training_gaze11 = utils.get_gaze_heatmap(training_gaze, 11)
		# gaze_coords_i, gaze_coords_j = training_gaze[i]

	with torch.no_grad():
		for i in range(len(training_inputs)):
			label = training_outputs[i]
			#print(inputs)
			#print(labels)
			traj_i, traj_j = training_inputs[i]
			traj_i = np.array(traj_i)
			traj_j = np.array(traj_j)
			traj_i = torch.from_numpy(traj_i).float().to(device)
			traj_j = torch.from_numpy(traj_j).float().to(device)

			if gaze_dropout:
				gaze26_i, gaze26_j = training_gaze26[i]
				# gaze11_i, gaze11_j = training_gaze11[i]
				# gaze_coords_i, gaze_coords_j = training_gaze[i]
				outputs, _, _, _ = reward_network.forward(traj_i, traj_j, gaze26_i, gaze26_j, train=False)
			else:
				outputs, _, _, _ = reward_network.forward(traj_i, traj_j, train=False)

			#forward to get logits
			
			#print(outputs)
			_, pred_label = torch.max(outputs,0)
			#print(pred_label)
			#print(label)
			if pred_label.item() == label:
				num_correct += 1.
	return num_correct / len(training_inputs)


def predict_reward_sequence(net, traj):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	rewards_from_obs = []
	with torch.no_grad():
		for s in traj:
			r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device), gaze26=None,train=False)[0].item()
			rewards_from_obs.append(r)
	return rewards_from_obs

def predict_traj_return(net, traj):
	return sum(predict_reward_sequence(net, traj))


if __name__=="__main__":
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
	parser.add_argument('--reward_model_path', default='', help="name and location for learned model params")
	parser.add_argument('--seed', default=0, help="random seed for experiments")
	parser.add_argument('--data_dir', help="where atari-head data is located")
	# parser.add_argument('--use_gaze', default=False, action='store_true', help="use gaze loss or not")
	parser.add_argument('--gaze_loss', default=None, type=str, help="type of gaze loss function: EMD, coverage, KD, None")
	parser.add_argument('--gaze_reg', default=0.01, help="gaze loss multiplier")
	parser.add_argument('--snippet_len', default=50, help="snippet lengths of trajectories used for training")
	parser.add_argument('--metric', default='rewards', help="metric to compare paired trajectories performance: rewards or returns")
	parser.add_argument('--mask_scores', default=False, action='store_true', help="mask scores on game screen or not")
	parser.add_argument('--use_old_data', default=False, action='store_true', help="use data without gaze from PPO checkpoints ")
	parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
	parser.add_argument('--gaze_dropout', default=False, action='store_true', help="use gaze modulated dropout or not")


	args = parser.parse_args()
	env_name = args.env_name
	if env_name == "spaceinvaders":
		env_id = "SpaceInvadersNoFrameskip-v4"
		agc_env_name =  "spaceinvaders"
	elif env_name == "mspacman":
		env_id = "MsPacmanNoFrameskip-v4"
		agc_env_name = "mspacman"
	elif env_name == "videopinball":
		env_id = "VideoPinballNoFrameskip-v4"
		agc_env_name = "pinball"
	elif env_name == "montezumarevenge":
		env_id = "MontezumaRevengeNoFrameskip-v4"
		agc_env_name = "revenge"
	elif env_name == "qbert":
		env_id = "QbertNoFrameskip-v4"
		agc_env_name = "qbert"
	elif env_name == "hero":
		env_id = "HeroNoFrameskip-v4"
		agc_env_name = "hero"
	elif env_name == "breakout":
		env_id = "BreakoutNoFrameskip-v4"
		agc_env_name = "breakout"
	elif env_name == "seaquest":
		env_id = "SeaquestNoFrameskip-v4"
		agc_env_name = "seaquest"
	elif env_name == "enduro":
		env_id = "EnduroNoFrameskip-v4"
		agc_env_name = "enduro"
	elif env_name == "pong":
		env_id = "PongNoFrameskip-v4"
		agc_env_name = "pong"
	else:
		print("env_name not supported")
		sys.exit(1)

	# use_gaze = args.use_gaze
	use_gaze = args.gaze_dropout or (args.gaze_loss is not None)
	gaze_loss_type = args.gaze_loss
	gaze_reg = float(args.gaze_reg)
	mask = args.mask_scores
	gaze_dropout = args.gaze_dropout

	env_type = "atari"
	print(env_type)
	#set seeds
	seed = int(args.seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)

	print("Training reward for", env_id)
	n_train = 6000 #10000 #number of pairs of trajectories to create
	snippet_length = int(args.snippet_len) #length of trajectory for training comparison
	lr = 0.0001
	weight_decay = 0.0
	num_iter = 5 #num times through training data
	l1_reg=0.0
	stochastic = True
	comp_metric = args.metric
	
	import os
	if not os.path.exists(args.reward_model_path):
		os.makedirs(args.reward_model_path)

	#env id, env type, num envs, and seed
	env = make_vec_env(env_id, 'atari', 1, seed,
					   wrapper_kwargs={
						   'clip_rewards':False,
						   'episode_life':False,
					   })

	env = VecFrameStack(env, 4)
	agent = PPO2Agent(env, env_type, stochastic)

	data_dir = args.data_dir
	# dataset = ds.AtariDataset(data_dir)
	# demonstrations, learning_returns = agc_demos.get_preprocessed_trajectories(agc_env_name, dataset, data_dir)
	dataset = ahd.AtariHeadDataset(env_name, data_dir)
	print('collected the raw demo data')
	demonstrations, learning_returns, learning_rewards, learning_gaze26 = utils.get_preprocessed_trajectories(env_name, dataset, data_dir, use_gaze, mask)
	print('processed the raw demo data')

	# Let's plot the returns to see if they are roughly monotonically increasing.
	#plt.plot(learning_returns)
	#plt.xlabel("Demonstration")
	#plt.ylabel("Return")
	#plt.savefig(env_type + "LearningCurvePPO.png")
	#plt.show()

	#sort the demonstrations according to ground truth reward

	print(len(learning_returns))
	print(len(demonstrations))
	# print([a[0] for a in zip(learning_returns, demonstrations)])
	#sort them based on human preferences
	# TODO: ******** why are we not sorting human demo trajectories??? ************
	# Ans: trajectory episodes already being sorted in 
	# demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

	# sorted_returns = sorted(learning_returns)
	# print(sorted_returns)
	#plt.plot(sorted_returns)
	#plt.show()

	# training_obs, training_labels, training_gaze = create_training_data(demonstrations, learning_returns, gaze_maps, n_train)
	training_data  = create_training_data(demonstrations, learning_returns, learning_rewards, learning_gaze26, n_train, use_gaze, snippet_length, comp_metric)
	training_obs, training_labels, training_gaze26 = training_data
	print("num training_obs", len(training_obs))
	print("num_labels", len(training_labels))


	# Get demos from stored PPO checkpoints
	# demonstrations2, learning_returns2, learning_rewards2 = novice.generate_novice_demos(env, env_name, agent, args.models_dir)
	# demonstrations2 = [x for _, x in sorted(zip(learning_returns2,demonstrations2), key=lambda pair: pair[0])]
	# sorted_returns2 = sorted(learning_returns2)
	# training_obs2, training_labels2 = novice.create_training_data(demonstrations2, n_train, snippet_length)
	# training_obs = training_obs + training_obs2u
	# training_labels = training_labels + training_labels2u
	# demonstrations = demonstrations + demonstrations2u
	# learning_returns = learning_returns + learning_returns2u
	# learning_rewards = learning_rewards + learning_rewards2

	# training_data = [training_obs, training_labels, training_gaze26, training_gaze11, training_gaze7]

	# Now we create a reward network and optimize it using the training data.
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	reward_net = Net(gaze_dropout, gaze_loss_type)
	reward_net.to(device)


	import torch.optim as optim
	optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
	learn_reward(reward_net, optimizer, training_data, num_iter, l1_reg, args.reward_model_path, gaze_loss_type, gaze_reg, gaze_dropout)

	with torch.no_grad():
		pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
	for i, p in enumerate(pred_returns):
		print(i,p,sum(learning_rewards[i]),learning_returns[i])

	print("accuracy", calc_accuracy(reward_net, training_obs, training_labels, training_gaze26, gaze_dropout))

	#TODO:add checkpoints to training process
	torch.save(reward_net.state_dict(), args.reward_model_path+"/model.pth")
