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

#cheat and sort them to see if it helps learning
#sorted_demos = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
#sorted_returns = sorted(learning_returns)
#print(sorted_returns)
#plt.plot(sorted_returns)


# Create training data by taking random 50 length crops of trajectories, computing the true returns and adding them to the training data with the correct label.
# def create_training_data(demonstrations, returns, gaze_maps, n_train):
def create_training_data(demonstrations, returns, rewards, gaze_maps, n_train, use_gaze, snippet_length):
    training_obs = []
    training_labels = []
    training_gaze = []
    num_demos = len(demonstrations)
    for n in range(n_train):
        ti, tj = 0, 0
        r_i, r_j = 0, 0   
        rew_i, rew_j = 0, 0
        
        #only add trajectories that are different returns
        # while(ti == tj):
        # while(r_i == r_j):
        while(rew_i == rew_j):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
            #print(ti, tj)
            #create random snippets
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
                gaze_i = gaze_maps[ti][ti_start:ti_start+snippet_length]
                gaze_j = gaze_maps[tj][tj_start:tj_start+snippet_length]


        # labels will be created differently
        # if r_i > r_j:
        if rew_i > rew_j:
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




class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(64, 1)


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        for x in traj:
            x = x.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            #x = x.view(-1, 1936)
            x = F.leaky_relu(self.fc1(x))
            #r = torch.tanh(self.fc2(x)) #clip reward?
            r = self.fc2(x)
            sum_rewards += r
            sum_abs_rewards += torch.abs(r)
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print(sum_rewards)
        return sum_rewards, sum_abs_rewards


    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        return torch.cat([cum_r_i, cum_r_j]), abs_r_i + abs_r_j


    def conv_map(self, traj):
        '''calculate cumulative return of trajectory'''
        # conv_map_traj = torch.zeros([len(traj), traj[0].shape[0], 7, 7], dtype=torch.float64)
        conv_map_traj = []
        for i,x in enumerate(traj):
            x = x.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x_final_conv = F.leaky_relu(self.conv4(x))
            # print(x_final_conv.shape)
            # x_final_conv = x_final_conv.squeeze()
            # print(x_final_conv.shape)
            x_final_conv = x_final_conv.sum(dim=1) #[batch size, 7, 7], summing all 16 conv filters
            # x = x.view(-1, 784)
            # x = F.leaky_relu(self.fc1(x))            
            # r = self.fc2(x)
            # print(x_final_conv.shape) # [7,7]
            # print(type(x_final_conv))
            # print(torch.min(x_final_conv))
            # print(torch.max(x_final_conv))
            min_x = torch.min(x_final_conv)
            max_x = torch.max(x_final_conv)
            x_norm = (x_final_conv - min_x)/(max_x - min_x)
            # print(x_norm)
            # conv_map_traj[i,:,:,:]=x_norm
            conv_map_traj.append(x_norm)

        conv_map_stacked = torch.stack(conv_map_traj)
        return conv_map_stacked


# In[111]:

def gaze_loss_KL(true_gaze, conv_gaze):
    loss = F.kl_div(true_gaze, conv_gaze)
    return loss

# wasserstein loss or Earth mover's distance
def gaze_loss_EMD(true_gaze, conv_gaze):
    from pyemd import emd, emd_samples
    from scipy.stats import wasserstein_distance

    # flatten input maps
    # print(type(true_gaze))
    maps = [img.ravel() for img in [true_gaze, conv_gaze]]

    # compute EMD using values
    d1 = emd_samples(maps[0], maps[1]) # 25.57794401220945
    d2 = wasserstein_distance(maps[0], maps[1]) # 25.76187896728515

    # compute EMD using distributions
    # N_BINS = 256
    # hists = [np.histogram(img, N_BINS, density=True)[0].astype(np.float64) for img in maps]

    # mgrid = np.meshgrid(np.arange(N_BINS), np.arange(N_BINS))
    # metric = np.abs(mgrid[0] - mgrid[1]).astype(np.float64)

    # emd(hists[0], hists[1], metric) # 25.862491463680065

    loss = d1
    return loss

# coverage based loss, only penalize if conv_gaze is not a superset of true_gaze
def gaze_loss_coverage(true_gaze, conv_gaze):
    true_gaze = true_gaze.cpu().detach().numpy()
    conv_gaze = conv_gaze.cpu().detach().numpy()

    # print(type(true_gaze))
    # flatten input maps
    maps = [img.ravel() for img in [true_gaze, conv_gaze]]

    # iterate over all coordinates of the true_gaze map
    for i in range(len(maps[0])):
        # compare pixel values between true and conv gaze map
        # add penalty if difference between true_gaze and conv_gaze pixel values is greater than a threshold (0.5)
        if abs(maps[0][i]-maps[1][i])>0.5:
            loss+=abs(maps[0][i]-maps[1][i])

    # normalize loss by batch size?

    return loss

# Now we train the network. I'm just going to do it one by one for now. Could adapt it for minibatches to get better gradients
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, training_gaze, num_iter, l1_reg, checkpoint_dir, use_gaze):

    # multiplier for gaze loss
    gaze_reg = 0.5

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
            labels = np.array([[training_labels[i]]])
            
            # print(traj_i)
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)
            
            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)

            output_loss = loss_criterion(outputs, labels)
            print('output loss: ', output_loss)

            if use_gaze:
                # ground truth human gaze maps (7x7)
                gaze_i, gaze_j = training_gaze[i]
                
                # print(len(gaze_i[0]))
                gaze_i = np.array(gaze_i)
                gaze_j = np.array(gaze_j)
                # print('GT gaze map shape: ', gaze_i.shape) #(50,1,7,7)
                # gaze_i = torch.from_numpy(gaze_i).float().to(device)
                # gaze_j = torch.from_numpy(gaze_j).float().to(device)

                # get normalized conv map output (7x7)
                gaze_map_i = reward_network.conv_map(traj_i).cpu().detach().numpy()
                gaze_map_j = reward_network.conv_map(traj_j).cpu().detach().numpy()
                # print('conv map shape: ', gaze_map_i.shape) #(50,1,7,7)

                gaze_loss = gaze_loss_EMD
                gaze_loss_i = gaze_loss(gaze_i, gaze_map_i)
                gaze_loss_j = gaze_loss(gaze_j, gaze_map_j)

                gaze_loss_total = (gaze_loss_i + gaze_loss_j)
                # gaze_loss_total = np.array([[gaze_loss_total]])
                # gaze_loss_total = torch.from_numpy(gaze_i).float().to(device)
                gaze_loss_total = torch.tensor(gaze_loss_total)
                print('gaze loss: ', gaze_loss_total)            
            
            if not use_gaze:
                loss = output_loss + l1_reg * abs_rewards
            else:
                loss = output_loss + l1_reg * abs_rewards + gaze_reg * gaze_loss_total
            print('total loss: ', loss)            

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



def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    num_correct = 0.
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

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
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
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
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
    parser.add_argument('--use_gaze', default=False, help="where atari-head data is located")
    parser.add_argument('--snippet_len', default=50, help="snippet lengths of trajectories used for training")

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
    else:
        print("env_name not supported")
        sys.exit(1)

    use_gaze = args.use_gaze

    env_type = "atari"
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    n_train = 10000 #number of pairs of trajectories to create
    snippet_length = int(args.snippet_len) #length of trajectory for training comparison
    lr = 0.0001
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg=0.0
    stochastic = True
    gaze_reg = 0.5

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
    demonstrations, learning_returns, learning_rewards, learning_gaze = utils.get_preprocessed_trajectories(env_name, dataset, data_dir, use_gaze)

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
    # demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    # sorted_returns = sorted(learning_returns)
    # print(sorted_returns)
    #plt.plot(sorted_returns)
    #plt.show()

    # training_obs, training_labels, training_gaze = create_training_data(demonstrations, learning_returns, gaze_maps, n_train)
    training_obs, training_labels, training_gaze = create_training_data(demonstrations, learning_returns, learning_rewards, learning_gaze, n_train, use_gaze, snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)

    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, training_gaze, num_iter, l1_reg, args.reward_model_path, use_gaze)

    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sum(learning_rewards[i]),learning_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))

    #TODO:add checkpoints to training process
    torch.save(reward_net.state_dict(), args.reward_model_path+"/model.pth")
