import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import get_gaze_heatmap

class Net(nn.Module):
	def __init__(self, gaze_dropout, gaze_loss_type):
		super().__init__()
		self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
		self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
		self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
		self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
		self.fc1 = nn.Linear(784, 64)
		#self.fc1 = nn.Linear(1936,64)
		self.fc2 = nn.Linear(64, 1)
		self.gaze_dropout = gaze_dropout
		self.gaze_loss_type = gaze_loss_type


	def cum_return(self, traj, gaze_coords, train):
		'''calculate cumulative return of trajectory'''
		sum_rewards = 0
		sum_abs_rewards = 0
		conv_map_traj = []
		conv_map_stacked = torch.tensor([[]])

		if self.gaze_dropout:
			gaze26 = get_gaze_heatmap(gaze_coords, 26)
			gaze11 = get_gaze_heatmap(gaze_coords, 11)	

		if self.gaze_loss_type is not None:
			gaze7 = get_gaze_heatmap(gaze_coords, 7)

		for i,x in enumerate(traj):
			x = x.permute(0,3,1,2) #get into NCHW format
			#compute forward pass of reward network
			x = F.leaky_relu(self.conv1(x))
			# print('conv1 shape',x.shape) # [1,16,26,26]

			# gaze modulated dropout
			if(self.gaze_dropout):
				assert(gaze26 is not None)
				x = self.gaze_modulated_dropout(x, gaze26[i], train)

			x = F.leaky_relu(self.conv2(x))
			# print('conv2 shape',x.shape) # [1,16,11,11]
			
			# gaze modulated dropout
			if(self.gaze_dropout):
				assert(gaze11 is not None)
				x = self.gaze_modulated_dropout(x, gaze11[i], train)
			
			
			x = F.leaky_relu(self.conv3(x))
			x_final_conv = F.leaky_relu(self.conv4(x))

			x = x_final_conv.view(-1, 784)
			#x = x.view(-1, 1936)
			x = F.leaky_relu(self.fc1(x))
			#r = torch.tanh(self.fc2(x)) #clip reward?
			r = self.fc2(x)
			sum_rewards += r
			sum_abs_rewards += torch.abs(r)

			# prepare conv map to be returned for gaze loss
			if self.gaze_loss_type is not None:
				assert(gaze7 is not None)
				# sum over all dimensions of the conv map
				conv_map = x_final_conv.sum(dim=1)

				# normalize the conv map
				min_x = torch.min(conv_map)
				max_x = torch.max(conv_map)
				x_norm = (conv_map - min_x)/(max_x - min_x)
				conv_map_traj.append(x_norm)
			
		##    y = self.scalar(torch.ones(1))
		##    sum_rewards += y
		if self.gaze_loss_type is not None:
			conv_map_stacked = torch.stack(conv_map_traj)
		return sum_rewards, sum_abs_rewards, conv_map_stacked


	def forward(self, traj_i, traj_j, gaze_coords_i=None, gaze_coords_j=None, train=False):
		'''compute cumulative return for each trajectory and return logits'''
		#print([self.cum_return(traj_i), self.cum_return(traj_j)])
		cum_r_i, abs_r_i, conv_map_i = self.cum_return(traj_i, gaze_coords_i, train)
		cum_r_j, abs_r_j, conv_map_j = self.cum_return(traj_j, gaze_coords_j, train)
		#print(abs_r_i + abs_r_j)
		return torch.cat([cum_r_i, cum_r_j]), abs_r_i + abs_r_j, conv_map_i, conv_map_j


	def conv_map(self, traj, gaze26, gaze11, train=False):
		'''calculate cumulative return of trajectory'''
		# conv_map_traj = torch.zeros([len(traj), traj[0].shape[0], 7, 7], dtype=torch.float64)
		conv_map_traj = []
		for i,x in enumerate(traj):
			x = x.permute(0,3,1,2) #get into NCHW format
			#compute forward pass of reward network
			x = F.leaky_relu(self.conv1(x))

			# gaze modulated dropout
			if(self.gaze_dropout):
				assert(gaze26 is not None)
				x = self.gaze_modulated_dropout(x, gaze26[i], train)			

			x = F.leaky_relu(self.conv2(x))
			
			# gaze modulated dropout
			if(self.gaze_dropout):
				assert(gaze11 is not None)
				x = self.gaze_modulated_dropout(x, gaze11[i], train)
			
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


	# gaze modulated dropout
	def gaze_modulated_dropout(self, conv_activation, gaze_map, train):
		# print('calling gaze modulated dropout')
		# print(gaze_map.shape) 
	
		# get the shape of the activations from previous layer
		act_shape = conv_activation.size()
		modulated_conv_act = torch.zeros(act_shape)

		dp = 0.7  # uniform dropout probability

		# Iterate over batch size
		# for i in range(act_shape[0]):
		# Iterate over each feature map
		# print('activation shape: ', act_shape) # [1,16,26,26]
		for j in range(act_shape[1]):
			
			# randomly sample array from a discrete uniform distribution between 0 & 1
			A = np.random.randint(1, size=(act_shape[2],act_shape[3])) #size=(2, 4)

			# convert numpy array to tensor
			device = torch.device('cuda:0')
			A = torch.from_numpy(A).float().to(device)


			# interpolate the gaze map to the shape of the activation
			# keep-probability mask K
			K = gaze_map
			# print(len(K), type(K)) # 50, list
			# print(K.shape)
			
			# Rescale the range of values in K to (1-dp,1) where dp is the dropout probability for uniform dropout
			# dp = 0.7 worked best for Chen et al.
			K = (K-(1-dp))/dp

			# Binary mask M=(K>A)
			K = torch.from_numpy(K).float().to(device)
			# print(K.shape, A.shape) # [20, 20], [20, 20]
			M = K>A
			M = M.float()

			curr_activation = conv_activation[:,j,:,:]		
			# curr_activation = curr_activation.squeeze()	
			# print(curr_activation.shape, M.shape, K.shape)

			# Apply the mask
			if train:
				# multiply with binary mask
				F = torch.bmm(curr_activation, M)
				# F = curr_activation*M

			else:
				# averaging effect at test time
				F = torch.bmm(curr_activation, K)
				# F = curr_activation*K

			# Normalize the features
			F = F/(1-dp)
			modulated_conv_act[:,j,:,:] = F
		
		# print(modulated_conv_act.type())
		return modulated_conv_act.to(device)
