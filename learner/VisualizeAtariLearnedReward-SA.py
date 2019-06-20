
# coding: utf-8

# In[1]:


import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
import matplotlib.pylab as plt
import argparse
import math, copy, time

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
parser.add_argument('--seed', default=0, help="random seed for experiments")
parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
parser.add_argument('--save_fig_dir', help ="where to save visualizations")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)


# In[3]:


#try it just for two trajectories
args = parser.parse_args()
env_name = args.env_name
save_fig_dir = args.save_fig_dir

if env_name == "spaceinvaders":
    env_id = "SpaceInvadersNoFrameskip-v4"
elif env_name == "mspacman":
    env_id = "MsPacmanNoFrameskip-v4"
elif env_name == "videopinball":
    env_id = "VideoPinballNoFrameskip-v4"
elif env_name == "beamrider":
    env_id = "BeamRiderNoFrameskip-v4"
else:
    env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
env_type = "atari"

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

print(env_id)

stochastic = True

reward_net_path = args.reward_net_path

#env id, env type, num envs, and seed
env = make_vec_env(env_id, 'atari', 1, 0,
                   wrapper_kwargs={
                       'clip_rewards':False,
                       'episode_life':False,
                   })


env = VecFrameStack(env, 4)
agent = PPO2Agent(env, env_type, stochastic)



import torch
import torch.nn as nn
import torch.nn.functional as F

# self attention
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #print('Attention')
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def positionalEncoding2D(input_tensor):
    #print(input_tensor.shape)
    batch_size = input_tensor.size()[0]
    # Attach 2D position layers to input tensor 
    kernel_w = input_tensor.size()[2]
    kernel_h = input_tensor.size()[3]       
    position_x = torch.arange(0., kernel_w).unsqueeze(0).cuda()
    position_y = torch.arange(0., kernel_h).unsqueeze(0).cuda()
    pe_x = torch.t(position_x.repeat(kernel_h,1).view(kernel_h,kernel_w)).unsqueeze(0)
    pe_y = position_y.repeat(1,kernel_w).view(kernel_w,kernel_h).unsqueeze(0)
    #print(pe_x.shape,pe_y.shape)
    att = torch.cat([pe_x, pe_y],0).unsqueeze(0)
    #print(att.shape)
    att = att.repeat(batch_size,1,1,1)
    #print(att.shape)
    
    out_tensor = torch.cat([input_tensor, att],1)
    #print( out_tensor.shape)
    return out_tensor

def flattenTensor(input_tensor):
    t_size = input_tensor.shape
    flat_input = torch.t(input_tensor.view(t_size[0], t_size[1]*t_size[2]))
    return flat_input


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))   



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h #h = 8; d_model=18
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        #print('MHA')
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # print(query.shape, key.shape, value.shape)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class FullyConnected(nn.Module):
    def __init__(self, nc=128, ndf=64, num_actions=1):
        super(FullyConnected, self).__init__()
        self.ndf = ndf
        self.max_pool = nn.MaxPool1d(ndf, return_indices=False)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(49, 64),
            # F.leaky_relu(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )


    def forward(self, input_state):
        #print('--> Action Prediction')
        #print('input:',input_state.shape)
        #import pdb;pdb.set_trace()
        #features = self.max_pool(input_state).view(input_state.shape[0],-1)
        features = self.max_pool(input_state).view(-1, 49)
        #print('features:',features.shape)
        output = self.main(features)
        #print('output:',output.shape)
        return output #.view(-1, 1).squeeze(1)

class ImageEncoder(nn.Module):
    "Process input RGB image (128x128)"
    def __init__(self, size, self_attn, feed_forward, dropout, nc=3, ndf=256, hn=30):
        super(ImageEncoder, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 
        
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(4, 16, 7, stride=3),
            # F.leaky_relu(),
            nn.LeakyReLU(0.2, inplace=True),                        
            nn.Conv2d(16, 16, 5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # F.leaky_relu(),       
            nn.Conv2d(16, 16, 3, stride=1),
            # nn.LeakyReLU(0.2, inplace=True),  
            # F.leaky_relu(),  
            #nn.Conv2d(ndf , ndf, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),         
            # nn.Conv2d(ndf, hn, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(16, 16, 3, stride=1),
            # nn.BatchNorm2d(hn),
            nn.LeakyReLU(0.2, inplace=True)
            # F.leaky_relu()
        )
   

    def forward(self, x):
        #import pdb;pdb.set_trace()
        images = self.main(x)
        # print('images:',images.shape)
        pencoded_is = positionalEncoding2D(images) 
        # print('pencoded_is:',pencoded_is.shape)
        flat_i = torch.flatten(pencoded_is,start_dim=2).permute(0,2,1)
        # print('flat_i:',flat_i.shape)
        x = self.sublayer[0](flat_i, lambda flat_i: self.self_attn(flat_i, flat_i, flat_i))
        output =  self.sublayer[1](x.squeeze(-1), self.feed_forward)
        return output

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        d_model=32 #? or 49?
        d_ff=256
        num_heads = 9#8
        embedding_channels = 18#32
        dropout=0.1
        nc=128
        ndf=64
        hn=30
        num_actions=1

        c = copy.deepcopy
        self.attn = MultiHeadedAttention(num_heads, embedding_channels)
        self.ff = PositionwiseFeedForward(embedding_channels, d_ff, dropout)

        self.model = nn.Sequential(ImageEncoder(embedding_channels, c(self.attn), c(self.ff), dropout, hn=hn), FullyConnected(nc=nc, ndf=ndf, num_actions=num_actions)).to(device)
        #self.model.apply(weights_init)
        

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        for x in traj:
            x = x.permute(0,3,1,2) #get into NCHW format
            r = self.model(x)
            sum_rewards += r
            sum_abs_rewards += torch.abs(r)
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        return torch.cat([cum_r_i, cum_r_j]), abs_r_i + abs_r_j




'''class Net(nn.Module):
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
        ## calculate cumulative return of trajectory
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
            r = torch.sigmoid(self.fc2(x))
            #r = self.fc2(x)
            sum_rewards += r
            sum_abs_rewards += torch.abs(r)
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print(sum_rewards)
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        ## compute cumulative return for each trajectory and return logits
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        return torch.cat([cum_r_i, cum_r_j]), abs_r_i + abs_r_j
'''

# In[5]:


reward = Net()
reward.load_state_dict(torch.load(reward_net_path))
reward.to(device)



#generate some trajectories for inspecting learned reward
checkpoint_min = 50
checkpoint_max = 600
checkpoint_step = 50

if env_name == "enduro":
    checkpoint_min = 3100
    checkpoint_max = 3650
elif env_name == "seaquest":
    checkpoint_min = 10
    checkpoint_max = 65
    checkpoint_step = 5

checkpoints_demos = []

for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints_demos.append('0000' + str(i))
        elif i < 100:
            checkpoints_demos.append('000' + str(i))
        elif i < 1000:
            checkpoints_demos.append('00' + str(i))
        elif i < 10000:
            checkpoints_demos.append('0' + str(i))
print(checkpoints_demos)



#generate some trajectories for inspecting learned reward
checkpoint_min = 650
checkpoint_max = 1450
checkpoint_step = 50
if env_name == "enduro":
    checkpoint_min = 3625
    checkpoint_max = 4425
    checkpoint_step = 50
elif env_name == "seaquest":
    checkpoint_min = 10
    checkpoint_max = 65
    checkpoint_step = 5
elif env_name == "hero":
    checkpoint_min = 100
    checkpoint_max = 2400
    checkpoint_step = 100
checkpoints_extrapolate = []
for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints_extrapolate.append('0000' + str(i))
        elif i < 100:
            checkpoints_extrapolate.append('000' + str(i))
        elif i < 1000:
            checkpoints_extrapolate.append('00' + str(i))
        elif i < 10000:
            checkpoints_extrapolate.append('0' + str(i))
print(checkpoints_extrapolate)


# In[9]:


from baselines.common.trex_utils import preprocess
model_dir = args.models_dir
demonstrations = []
learning_returns_demos = []
pred_returns_demos = []
for checkpoint in checkpoints_demos:

    model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
    if env_name == "seaquest":
        model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

    agent.load(model_path)
    episode_count = 1
    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            #print(ob.shape)
            traj.append(preprocess(ob, env_name))
            steps += 1
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                break
        print("traj length", len(traj))
        print("demo length", len(demonstrations))

        demonstrations.append(traj)
        learning_returns_demos.append(acc_reward)
        pred_returns_demos.append(reward.cum_return(torch.from_numpy(np.array(traj)).float().to(device))[0].item())
        print("pred return", pred_returns_demos[-1])

learning_returns_extrapolate = []
pred_returns_extrapolate = []


'''
for checkpoint in checkpoints_extrapolate:

    model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
    if env_name == "seaquest":
        model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

    agent.load(model_path)
    if env_name == "enduro":
        episode_count = 1
    else:
        episode_count = 3
    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            #print(ob.shape)
            traj.append(preprocess(ob, env_name))
            steps += 1
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                break
        print("traj length", len(traj))
        print("demo length", len(demonstrations))
        demonstrations.append(traj)
        learning_returns_extrapolate.append(acc_reward)
        pred_returns_extrapolate.append(reward.cum_return(torch.from_numpy(np.array(traj)).float().to(device))[0].item())
        print("pred return", pred_returns_extrapolate[-1])

'''
env.close()






# In[10]:


def convert_range(x,minimum, maximum,a,b):
    return (x - minimum)/(maximum - minimum) * (b - a) + a


# In[12]:


buffer = 20
if env_name == "pong":
    buffer = 2
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         # 'figure.figsize': (6, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
learning_returns_all = learning_returns_demos + learning_returns_extrapolate
pred_returns_all = pred_returns_demos + pred_returns_extrapolate
print(pred_returns_all)
print(learning_returns_all)
plt.plot(learning_returns_extrapolate, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_extrapolate],'bo')
plt.plot(learning_returns_demos, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_demos],'ro')
plt.plot([min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer],[min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer],'g--')
plt.plot([min(0, min(learning_returns_all)-2),max(learning_returns_demos)],[min(0, min(learning_returns_all)-2),max(learning_returns_demos)],'k-', linewidth=2)
plt.axis([min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer,min(0, min(learning_returns_all)-2),max(learning_returns_all)+buffer])
plt.xlabel("Ground Truth Returns")
plt.ylabel("Predicted Returns (normalized)")
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "_gt_vs_pred_rewards.png")

#plt.axis('square')


# In[30]:


print(learning_returns_all)
returns_to_plot = sorted([np.random.choice(learning_returns_all) for _ in range(3)])
demos_to_plot = []
for r in returns_to_plot:
    print("searching for", r)
    for i,d in enumerate(demonstrations):
        if learning_returns_all[i] == r:
            print(learning_returns_all[i])
            demos_to_plot.append(d)
            break
print(returns_to_plot)
print(len(demos_to_plot))



# import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'xx-large',
#           #'figure.figsize': (15, 5),
#          'axes.labelsize': 'xx-large',
#          'axes.titlesize':'xx-large',
#          'xtick.labelsize':'xx-large',
#          'ytick.labelsize':'xx-large'}
# pylab.rcParams.update(params)
#print out the actual time series of rewards predicted by nnet for each trajectory.
cnt = 0
with torch.no_grad():
    d = demos_to_plot[0]
    plt.figure(2)
    rewards = []
    print(cnt)
    cnt += 1
    for s in d:
        r = reward.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
        #print(r)
        rewards.append(r)
    plt.ylabel("reward")
    plt.plot(rewards[2:-1])
    plt.xlabel("time")

    plt.title("GT Return = {}".format(returns_to_plot[0]))

#plt.savefig("learned_mcar_return.png")
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "_" + str(returns_to_plot[0]) + "_RewardPlots.png")

with torch.no_grad():
    d = demos_to_plot[1]
    plt.figure(3)
    rewards = []
    print(cnt)
    cnt += 1
    for s in d:
        r = reward.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
        #print(r)
        rewards.append(r)
    plt.ylabel("reward")
    plt.plot(rewards[2:-1])
    plt.xlabel("time")

    plt.title("GT Return = {}".format(returns_to_plot[1]))

#plt.savefig("learned_mcar_return.png")
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "_" + str(returns_to_plot[1]) + "_RewardPlots.png")


with torch.no_grad():
    d = demos_to_plot[2]
    plt.figure(4)
    rewards = []
    print(cnt)
    cnt += 1
    for s in d:
        r = reward.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
        #print(r)
        rewards.append(r)
    plt.ylabel("reward")
    plt.plot(rewards[2:-1])
    plt.xlabel("time")

    plt.title("GT Return = {}".format(returns_to_plot[2]))

#plt.savefig("learned_mcar_return.png")
    plt.tight_layout()
    plt.savefig(save_fig_dir + "/" + env_name + "_" + str(returns_to_plot[2]) + "_RewardPlots.png")

#plt.show()



min_reward = 100000
max_reward = -100000
cnt = 0
with torch.no_grad():
    for d in demonstrations:
        print(cnt)
        cnt += 1
        for i,s in enumerate(d[2:-1]):
            r = reward.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            if r < min_reward:
                min_reward = r
                min_frame = s
                min_frame_i = i+2
            elif r > max_reward:
                max_reward = r
                max_frame = s
                max_frame_i = i+2






def mask_coord(i,j,frames, mask_size, channel):
    #takes in i,j pixel and stacked frames to mask
    masked = frames.copy()
    masked[:,i:i+mask_size,j:j+mask_size,channel] = 0
    return masked

def gen_attention_maps(frames, mask_size):

    orig_frame = frames

    #okay so I want to vizualize what makes these better or worse.
    _,height,width,channels = orig_frame.shape

    #find reward without any masking once
    r_before = reward.cum_return(torch.from_numpy(np.array([orig_frame])).float().to(device))[0].item()
    heat_maps = []
    for c in range(4): #four stacked frame channels
        delta_heat = np.zeros((height, width))
        for i in range(height-mask_size):
            for j in range(width - mask_size):
                #get masked frames
                masked_ij = mask_coord(i,j,orig_frame, mask_size, c)
                r_after = r = reward.cum_return(torch.from_numpy(np.array([masked_ij])).float().to(device))[0].item()
                r_delta = abs(r_after - r_before)
                #save to heatmap
                delta_heat[i:i+mask_size, j:j+mask_size] += r_delta
        heat_maps.append(delta_heat)
    return heat_maps



#plot heatmap
mask_size = 3
delta_heat_max = gen_attention_maps(max_frame, mask_size)
delta_heat_min = gen_attention_maps(min_frame, mask_size)


# In[45]:


plt.figure(5)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_max[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_attention.png", bbox_inches='tight')
#plt.show()
#plt.title("max frame")
#plt.savefig("/home/dsbrown/Pictures/scott_berkeley/" + env_name + "_attention_maxframes.png", bbox_inches='tight')


# In[40]:

plt.figure(6)
print(max_frame_i)
print(max_reward)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(max_frame[0][:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_frames.png", bbox_inches='tight')
#plt.savefig("/home/dsbrown/Pictures/scott_berkeley/" + env_name + "_maxframes.png", bbox_inches='tight')
#plt.savefig("/home/dsbrown/Code/learning-rewards-of-learners/learner/figs/" + env_name + "_maxreward_obs.png")
#plt.figure(2)
#plt.imshow(demonstrations[0][max_frame_i-5][0][:,:,0])
#plt.show()


# In[46]:

plt.figure(7)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_min[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_attention.png", bbox_inches='tight')
#plt.title("min frame")
#plt.savefig("/home/dsbrown/Pictures/scott_berkeley/" + env_name + "_attention_minframes.png", bbox_inches='tight')


# In[42]:


print(min_frame_i)
print(min_reward)
plt.figure(8)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(min_frame[0][:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_frames.png", bbox_inches='tight')
#plt.savefig("/home/dsbrown/Pictures/scott_berkeley/" + env_name + "_minframes.png", bbox_inches='tight')
#plt.savefig("/home/dsbrown/Code/learning-rewards-of-learners/learner/figs/" + env_name + "_minreward_obs.png")
#plt.figure(2)
#plt.imshow(demonstrations[0][max_frame_i-5][0][:,:,0])
#plt.show()


# In[54]:


#random frame heatmap
d_rand = np.random.randint(len(demonstrations))
f_rand = np.random.randint(len(demonstrations[d_rand]))
rand_frames = demonstrations[d_rand][f_rand]


# In[55]:

plt.figure(9)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(rand_frames[0][:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "random_frames.png", bbox_inches='tight')
#plt.savefig("/home/dsbrown/Pictures/scott_berkeley/" + env_name + "_randframes.png", bbox_inches='tight')
#plt.savefig("/home/dsbrown/Code/learning-rewards-of-learners/learner/figs/" + env_name + "_minreward_obs.png")
#plt.figure(2)
#plt.imshow(demonstrations[0][max_frame_i-5][0][:,:,0])
#plt.show()


# In[56]:


delta_heat_rand = gen_attention_maps(rand_frames, mask_size)
plt.figure(10)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_rand[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
#plt.colorbar()
plt.savefig(save_fig_dir + "/" + env_name + "random_attention.png", bbox_inches='tight')
#plt.title("max frame")
#plt.savefig("/home/dsbrown/Pictures/scott_berkeley/" + env_name + "_attention_randframes.png", bbox_inches='tight')
