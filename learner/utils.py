import numpy as np
import cv2
import csv
import os
import torch
from os import path, listdir
import gaze_heatmap as gh
import time

cv2.ocl.setUseOpenCL(False)

def normalize_state(obs):
    return obs / 255.0

def normalize(obs, max_val):
    #TODO: discard frames with no gaze
    if(max_val!=0):
        norm_map = obs/float(max_val)
    else:
        norm_map = obs
    return norm_map

def mask_score(obs, crop_top = True):
    if crop_top:
        #takes a stack of four observations and blacks out (sets to zero) top n rows
        n = 10
        #no_score_obs = copy.deepcopy(obs)
        obs[:,:n,:,:] = 0
    else:
        n = 20
        obs[:,-n:,:,:] = 0
    return obs

#need to grayscale and warp to 84x84
def GrayScaleWarpImage(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width=84
    height=84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    #frame = np.expand_dims(frame, -1)
    return frame

def MaxSkipAndWarpFrames(trajectory_dir, frames):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    # num_frames = len(listdir(trajectory_dir))
    num_frames = len(frames)
    # print('total images:', num_frames)
    skip=4

    sample_pic = np.random.choice(listdir(trajectory_dir))
    image_path = path.join(trajectory_dir, sample_pic)
    pic = cv2.imread(image_path)
    obs_buffer = np.zeros((2,)+pic.shape, dtype=np.uint8)
    max_frames = []
    for i in range(num_frames):
        #TODO: check that i should max before warping.
        # b = trajectory_dir.split("_")
        # img_name =  "_".join(b[1:3]) + "_" + str(i) + ".png"
        img_name = frames[i] + ".png"

        if i % skip == skip - 2:
            # print(path.join(trajectory_dir, img_name))
            obs = cv2.imread(path.join(trajectory_dir, img_name))
            
            # print(type(obs))
            obs_buffer[0] = obs
        if i % skip == skip - 1:
            # print(path.join(trajectory_dir, img_name))
            obs = cv2.imread(path.join(trajectory_dir, img_name))
            obs_buffer[1] = obs
            # if(i==3):
            #     print(path.join(trajectory_dir, img_name))
            #     cv2.imshow("obs",obs)
            #     cv2.waitKey(0)
            #warp max to 80x80 grayscale
            image = obs_buffer.max(axis=0)
            warped = GrayScaleWarpImage(image)
            max_frames.append(warped)
    # print('num img frames: ', len(max_frames))
    return max_frames

def StackFrames(frames):
    import copy
    """stack every four frames to make an observation (84,84,4)"""
    stacked = []
    stacked_obs = np.zeros((84,84,4))
    for i in range(len(frames)):
        # if(i==3):
            # print(path.join(trajectory_dir, img_name))
            # cv2.imshow("obs",frames[i])
            # cv2.waitKey(0)
        if i >= 3:
            stacked_obs[:,:,0] = frames[i-3]
            stacked_obs[:,:,1] = frames[i-2]
            stacked_obs[:,:,2] = frames[i-1]
            stacked_obs[:,:,3] = frames[i]
            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0))
    return stacked


def CreateGazeMap(gaze_coords, pic):
    import math
    w, h = 7, 7
    old_h, old_w = pic.shape[0], pic.shape[1]
    obs = np.zeros((w, h))
    # print(gaze_coords)
    gaze_freq = {}
    if(not np.isnan(gaze_coords).all()):      
        for j in range(0,len(gaze_coords),2):
            if(not np.isnan(gaze_coords[j]) and not np.isnan(gaze_coords[j+1])):
                x = (gaze_coords[j])*w/old_w
                y = (gaze_coords[j+1])*h/old_h

                # g_coords = [gaze_coords[j], gaze_coords[j+1]]
                # i_size = [old_w,old_h]
                # print('orig size: ',old_w,old_h)
                # print('gaze_coords:',gaze_coords[j], gaze_coords[j+1])
                # print('coords for 84x84: ',x,y)
                #if(gaze_coords[j]>old_w-1 or gaze_coords[j+1]>old_h-1):
                #     print('gaze coordinates outside right/lower border')
                #     print('gaze coordinates: ',gaze_coords[j], gaze_coords[j+1])
                #    print(str(g_coords)+'\t'+str(i_size))
                #     print('image size: ', old_w, old_h)
                    # exit(1)
                # if(gaze_coords[j]<0 or gaze_coords[j+1]<0):
                    # print('gaze coordinates outside left/upper border')
                    # print('gaze coordinates: ', gaze_coords[j], gaze_coords[j+1])
                    # print('image size: ', old_w, old_h)
                    
                    # print(str(g_coords)+'\t'+str(i_size))
                    # exit(1)
                x, y = min(int(x),w-1), min(int(y),h-1)
                
                # print('int',x,y)
                if (x,y) not in gaze_freq:
                    gaze_freq[(x,y)] = 1
                else:
                    gaze_freq[(x,y)] += 1
    
    # Create the gaze mask based on how frequently a coordinate is fixated upon
    for coords in gaze_freq:
        x, y = coords
        # print(x)
        obs[y,x] = gaze_freq[coords]

    if np.isnan(obs).any():
        print('nan gaze map created')
        exit(1)

    return obs

def MaxSkipGaze(gaze, heatmap_size):
    """take a list of gaze coordinates and max over every 3rd and 4th observation"""
    num_frames = len(gaze)
    # print('total gaze items: ', num_frames)
    skip=4
    # sample_pic = np.random.choice(listdir(trajectory_dir))
    # image_path = path.join(trajectory_dir, sample_pic)
    # pic = cv2.imread(image_path)
    # pic_small = cv2.resize(pic, (heatmap_size, heatmap_size), interpolation=cv2.INTER_AREA)
    # pic_small = cv2.cvtColor(pic_small, cv2.COLOR_BGR2GRAY)
    obs_buffer = np.zeros((2,)+(heatmap_size,heatmap_size), dtype=np.float32)
    max_frames = []
    for i in range(num_frames):
        g = gaze[i]
        g = np.squeeze(g)
        if i % skip == skip - 2:
            # obs = CreateGazeMap(g, pic)
            obs_buffer[0] = g
        if i % skip == skip - 1:
            # obs = CreateGazeMap(g, pic)
            obs_buffer[1] = g
            image = obs_buffer.max(axis=0)
            max_frames.append(image)
    # print('num gaze frames: ', len(max_frames))
    if np.isnan(max_frames).any():
        print('nan max gaze map created')
        exit(1)
            
    return max_frames


def MaxGazeHeatmaps(gaze_hm_1, gaze_hm_2, heatmap_size):
    '''takes 2 stacks of 4 heatmaps for an observation (corresponding to 3rd and 4th frame) and take the max of the two'''
    obs_buffer = np.zeros((2,)+(heatmap_size,heatmap_size), dtype=np.float32)
    max_frames = []

    for gh_i,gh_j in zip(gaze_hm_1, gaze_hm_2):
        # g = gaze[i]
        # g = np.squeeze(g)
        # if i % skip == skip - 2:
            # obs = CreateGazeMap(g, pic)
        obs_buffer[0] = gh_i
        # if i % skip == skip - 1:
            # obs = CreateGazeMap(g, pic)
        obs_buffer[1] = gh_j
        image = obs_buffer.max(axis=0)
        max_frames.append(image)

    return max_frames

def SkipGazeCoords(gaze_coords):
    """take a list of gaze coordinates and uses every 3rd and 4th observation"""
    num_frames = len(gaze_coords)
    # print('total gaze items: ', num_frames)
    skip=4
    
    skipped_frames = []
    obs_buffer = []
    for i in range(num_frames):
        g = gaze_coords[i]
        # g = np.squeeze(g)
        
        if i % skip == skip - 2:
            # obs = CreateGazeMap(g, pic)
            obs_buffer.append(g)
            # print('len(obs_buffer1): ',len(obs_buffer))
        if i % skip == skip - 1:
            # obs = CreateGazeMap(g, pic)
            # obs_buffer[1] = g
            obs_buffer.append(g)
            # image = obs_buffer.max(axis=0)
            # print('len(obs_buffer): ',len(obs_buffer))
            skipped_frames.append(obs_buffer)
            obs_buffer = []
    
    return skipped_frames

def CollapseGaze(gaze_frames, heatmap_size):
    import copy
    """combine every four frames to make an observation (84,84)"""
    stacked = []
    stacked_obs = np.zeros((heatmap_size,heatmap_size))
    for i in range(len(gaze_frames)):
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs = gaze_frames[i-3]
            stacked_obs = stacked_obs + gaze_frames[i-2]
            stacked_obs = stacked_obs + gaze_frames[i-1]
            stacked_obs = stacked_obs + gaze_frames[i]

            # Normalize the gaze mask
            max_gaze_freq = np.amax(stacked_obs)
            stacked_obs = normalize(stacked_obs, max_gaze_freq)

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0)) # shape: (1,7,7)

    # if np.isnan(stacked).any():
    #     print('nan stacked gaze map created')
    #     exit(1)
    return stacked

def CollapseGazeHeatmaps(maxed_gaze, heatmap_size):
    import copy
    """combine four frames to make an observation (84,84)"""
    # stacked = []
    stacked_obs = np.zeros((heatmap_size,heatmap_size))
    for i in range(len(maxed_gaze)):
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs = maxed_gaze[i-3]
            stacked_obs = stacked_obs + maxed_gaze[i-2]
            stacked_obs = stacked_obs + maxed_gaze[i-1]
            stacked_obs = stacked_obs + maxed_gaze[i]

            # Normalize the gaze mask
            max_gaze_freq = np.amax(stacked_obs)
            stacked_obs = normalize(stacked_obs, max_gaze_freq)

            # stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0)) # shape: (1,7,7)

    # if np.isnan(stacked).any():
    #     print('nan stacked gaze map created')
    #     exit(1)
    return np.expand_dims(copy.deepcopy(stacked_obs),0) #(1,7,7)

def StackGaze(gaze_frames, heatmap_size):
    import copy
    """combine every four frames to make an observation (84,84)"""
    stacked = []
    # stacked_obs = np.zeros((7,7))
    stacked_obs = np.zeros((heatmap_size,heatmap_size,4))
    for i in range(len(gaze_frames)):
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs[:,:,0] = gaze_frames[i-3]
            stacked_obs[:,:,1] = gaze_frames[i-2]
            stacked_obs[:,:,2] = gaze_frames[i-1]
            stacked_obs[:,:,3] = gaze_frames[i]

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0)) # shape: (1,7,7)

    # if np.isnan(stacked).any():
    #     print('nan stacked gaze map created')
    #     exit(1)
    return stacked

def StackGazeCoords(gaze_coord_pairs):
    import copy
    """combine every four coordinate pairs to make an observation"""
    stacked = []
    # stacked_obs = np.zeros((7,7))
    # stacked_obs = np.zeros((heatmap_size,heatmap_size,4))
    for i in range(len(gaze_coord_pairs)):
        stacked_obs = []
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs.append(gaze_coord_pairs[i-3])
            stacked_obs.append(gaze_coord_pairs[i-2])
            stacked_obs.append(gaze_coord_pairs[i-1])
            stacked_obs.append(gaze_coord_pairs[i])

            # stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0)) # shape: (1,7,7)
            stacked.append(stacked_obs)

    # if np.isnan(stacked).any():
    #     print('nan stacked gaze map created')
    #     exit(1)
    return stacked

def MaxSkipReward(rewards):
    """take a list of rewards and max over every 3rd and 4th observation"""
    num_frames = len(rewards)
    skip=4
    max_frames = []
    obs_buffer = np.zeros((2,))
    for i in range(num_frames):
        r = rewards[i]
        if i % skip == skip - 2:
            
            obs_buffer[0] = r
        if i % skip == skip - 1:
            
            obs_buffer[1] = r
            rew = obs_buffer.max(axis=0)
            max_frames.append(rew)
    # print('num reward frames: ', len(max_frames))
    return max_frames


def StackReward(rewards):
    import copy
    """combine every four frames to make an observation"""
    stacked = []
    stacked_obs = np.zeros((1,))
    for i in range(len(rewards)):
        if i >= 3:
            # Sum over the rewards across four frames
            stacked_obs = rewards[i-3]
            stacked_obs = stacked_obs + rewards[i-2]
            stacked_obs = stacked_obs + rewards[i-1]
            stacked_obs = stacked_obs + rewards[i]

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0))
    return stacked

def get_sorted_traj_indices(env_name, dataset):
    #need to pick out a subset of demonstrations based on desired performance
    #first let's sort the demos by performance, we can use the trajectory number to index into the demos so just
    #need to sort indices based on 'score'
    game = env_name
    #Note, I'm also going to try only keeping the full demonstrations that end in terminal
    traj_indices = []
    traj_scores = []
    traj_dirs = []
    traj_rewards = []
    traj_gaze = []
    traj_frames = []
    print('traj length: ',len(dataset.trajectories[game]))
    for t in dataset.trajectories[game]:
        # if env_name == "revenge":
        #     traj_indices.append(t)
        #     traj_scores.append(dataset.trajectories[g][t][-1]['score'])

        # elif dataset.trajectories[g][t][-1]['terminal']:
        traj_indices.append(t)
        traj_scores.append(dataset.trajectories[game][t][-1]['score'])
        traj_dirs.append(dataset.trajectories[game][t][-1]['img_dir'])
        traj_rewards.append([dataset.trajectories[game][t][i]['reward'] for i in range(len(dataset.trajectories[game][t]))])
        traj_gaze.append([dataset.trajectories[game][t][i]['gaze_positions'] for i in range(len(dataset.trajectories[game][t]))])
        traj_frames.append([dataset.trajectories[game][t][i]['frame'] for i in range(len(dataset.trajectories[game][t]))])

    sorted_traj_indices = [x for _, x in sorted(zip(traj_scores, traj_indices), key=lambda pair: pair[0])]
    sorted_traj_scores = sorted(traj_scores)
    sorted_traj_dirs = [x for _, x in sorted(zip(traj_scores, traj_dirs), key=lambda pair: pair[0])]
    sorted_traj_rewards = [x for _, x in sorted(zip(traj_scores, traj_rewards), key=lambda pair: pair[0])]
    sorted_traj_gaze = [x for _, x in sorted(zip(traj_scores, traj_gaze), key=lambda pair: pair[0])]
    sorted_traj_frames = [x for _, x in sorted(zip(traj_scores, traj_frames), key=lambda pair: pair[0])]

    #print(sorted_traj_scores)
    #print(len(sorted_traj_scores))
    print("Max human score", max(sorted_traj_scores))
    print("Min human score", min(sorted_traj_scores))
    # print(len(sorted_traj_scores), len(sorted_traj_indices), \
    # len(sorted_traj_dirs), len(sorted_traj_rewards), len(sorted_traj_gaze))

    #so how do we want to get demos? how many do we have if we remove duplicates?
    seen_scores = set()
    non_duplicates = []
    for i,s,d,r,g,f in zip(sorted_traj_indices, sorted_traj_scores, sorted_traj_dirs, sorted_traj_rewards, sorted_traj_gaze, sorted_traj_frames):
        # print('s: ',s)
        if s not in seen_scores:
            seen_scores.add(s)
            non_duplicates.append((i,s,d,r,g,f))
    print("num non duplicate scores", len(seen_scores))
    if env_name == "spaceinvaders":
        start = 0
        skip = 3
    elif env_name == "revenge":
        start = 0
        skip = 1
    elif env_name == "qbert":
        start = 0
        skip = 3
    elif env_name == "mspacman":
        start = 0
        skip = 1
    else:   # TODO: confirm best logic for all games
        start = 0
        skip = 3
    num_demos = 12
    # demos = non_duplicates[start:num_demos*skip + start:skip] 
    demos = non_duplicates # don't skip any demos
    #print("(index, score) pairs:",demos)
    return demos


def get_preprocessed_trajectories(env_name, dataset, data_dir, use_gaze, mask_scores):
    """returns an array of trajectories corresponding to what you would get running checkpoints from PPO
       demonstrations are grayscaled, maxpooled, stacks of 4 with normalized values between 0 and 1 and
       top section of screen is masked
    """
    #mspacman score is on the bottom of the screen
    if env_name == 'mspacman':
        crop_top = False
    else:
        crop_top = True

    demos = get_sorted_traj_indices(env_name, dataset)
    human_scores = []
    human_demos = []
    human_rewards = []
    human_gaze = []
    # human_gaze_26 = []
    # human_gaze_11 = []
    # human_gaze_7 = []

    # img_frames = []
    print('len demos: ',len(demos))
    for indx, score, img_dir, rew, gaze, frame in demos:
        human_scores.append(score)

        # traj_dir = path.join(data_dir, 'screens', env_name, str(indx))
        traj_dir = path.join(data_dir, env_name, img_dir)
        maxed_traj = MaxSkipAndWarpFrames(traj_dir, frame)
        stacked_traj = StackFrames(maxed_traj)

        demo_norm_mask = []
        #normalize values to be between 0 and 1 and have top part masked
        for ob in stacked_traj:
            if mask_scores:
                demo_norm_mask.append(mask_score(normalize_state(ob), crop_top))
            else:
                demo_norm_mask.append(normalize_state(ob))  # currently not cropping
        human_demos.append(demo_norm_mask)

        # skip and stack reward
        maxed_reward = MaxSkipReward(rew)
        stacked_reward = StackReward(maxed_reward)      
        human_rewards.append(stacked_reward)

        if(use_gaze):
            # just return the gaze coordinates themselves
            # skipped_gaze = SkipGazeCoords(gaze)
            # stacked_gaze = StackGazeCoords(skipped_gaze)
            # human_gaze.append(stacked_gaze)

            # generate gaze heatmaps as per Ruohan's algorithm
            h = gh.DatasetWithHeatmap()
            g_26 = h.createGazeHeatmap(gaze, 26)
            # g_11 = h.createGazeHeatmap(gaze, 11)
            # g_7 = h.createGazeHeatmap(gaze, 7)

            # print('g type: ', type(g_11))

            # skip and stack gaze
            maxed_gaze_26 = MaxSkipGaze(g_26, 26)
            stacked_gaze_26 = CollapseGaze(maxed_gaze_26, 26)
            human_gaze_26.append(stacked_gaze_26)

            # maxed_gaze_11 = MaxSkipGaze(g_11, 11)
            # stacked_gaze_11 = CollapseGaze(maxed_gaze_11, 11)
            # human_gaze_11.append(stacked_gaze_11)

            # maxed_gaze_7 = MaxSkipGaze(g_7, 7)
            # stacked_gaze_7 = CollapseGaze(maxed_gaze_7, 7)
            # human_gaze_7.append(stacked_gaze_7)

            # print('maxed gaze type: ',type(maxed_gaze_11)) #list
            # print('stacked gaze type: ',type(stacked_gaze_11)) #list

    if(use_gaze):    
        print(len(human_demos[0]), len(human_rewards[0]), len(human_gaze[0]))
        print(len(human_demos), len(human_rewards), len(human_gaze))
    return human_demos, human_scores, human_rewards, human_gaze_26


def read_gaze_file(game_file):
    with open(game_file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines] 
    return lines

def get_gaze_heatmap(gaze_coords, heatmap_size):
    '''generate gaze heatmps of specific size for an entire trajectory'''
    print('generating gaze heatmap for trajectory of size: ', heatmap_size)
    human_gaze = []

    # for each observation in the trajectory
    for gaze in gaze_coords:
    # generating heatmap for a single frame stack (a list of 4 gaze coord pairs)
        h = gh.DatasetWithHeatmap()
        # print('length of gaze: ',len(gaze), len(gaze[0]))
        # print('gaze[0]:', gaze[0])
        gaze_hm_1 = h.createGazeHeatmap([g[0] for g in gaze], heatmap_size)
        gaze_hm_2 = h.createGazeHeatmap([g[1] for g in gaze], heatmap_size)

        # skip and stack gaze
        maxed_gaze = MaxGazeHeatmaps(gaze_hm_1.squeeze(), gaze_hm_2.squeeze(), heatmap_size)
        stacked_gaze = CollapseGazeHeatmaps(maxed_gaze, heatmap_size)
        human_gaze.append(stacked_gaze)

    return human_gaze


def get_all_gaze_heatmaps(gaze_coords, heatmap_size):
    '''generate gaze heatmps of specific size for an all trajectories'''
    print('generating gaze heatmap of size: ', heatmap_size)
    human_gaze = [[],[]]
    i = 0 
    # for each demo pair
    print('no of demo pairs: ',len(gaze_coords))
    for demo_gaze_coords in gaze_coords:
        print(i)
        i += 1
        # for each observation in the trajectory
        # print('trajectory len: ',len(traj_gaze_coords[0]), len(traj_gaze_coords[1]))
        traj_i_gaze, traj_j_gaze = [], []
        traj_i, traj_j = demo_gaze_coords[0], demo_gaze_coords[1]
        # print('gaze coords in traj_i: ',len(traj_i))

        start = time.time()
        for gaze in traj_i:
            # generating heatmap for a single frame stack (a list of 4 gaze coord pairs)
            h = gh.DatasetWithHeatmap()
            # print('length of gaze: ',len(gaze), len(gaze[0]))
            # print('gaze[0]:', gaze[0])
            # s = time.time()
            gaze_hm_1 = h.createGazeHeatmap([g[0] for g in gaze], heatmap_size)
            # e = time.time()
            # print('time for a stack of 4 frames: ', e-s)
            gaze_hm_2 = h.createGazeHeatmap([g[1] for g in gaze], heatmap_size)

            # skip and stack gaze
            # s = time.time()
            maxed_gaze = MaxGazeHeatmaps(gaze_hm_1.squeeze(), gaze_hm_2.squeeze(), heatmap_size)
            # e = time.time()
            # print('time for maxing over 3rd/4th frame: ', e-s)

            # s = time.time()
            stacked_gaze = CollapseGazeHeatmaps(maxed_gaze, heatmap_size)
            # e = time.time()
            # print('stacking frame time: ', e-s)

            traj_i_gaze.append(stacked_gaze)
        human_gaze[0].append(traj_i_gaze)
        end = time.time()
        print('time for 50 heatmaps: ', end-start)

        start = time.time()
        for gaze in traj_j:
            # generating heatmap for a single frame stack (a list of 4 gaze coord pairs)
            h = gh.DatasetWithHeatmap()
            # print('length of gaze: ',len(gaze), len(gaze[0]))
            # print('gaze[0]:', gaze[0])
            gaze_hm_1 = h.createGazeHeatmap([g[0] for g in gaze], heatmap_size)
            gaze_hm_2 = h.createGazeHeatmap([g[1] for g in gaze], heatmap_size)

            # skip and stack gaze
            maxed_gaze = MaxGazeHeatmaps(gaze_hm_1.squeeze(), gaze_hm_2.squeeze(), heatmap_size)
            stacked_gaze = CollapseGazeHeatmaps(maxed_gaze, heatmap_size)
            traj_j_gaze.append(stacked_gaze)
        human_gaze[1].append(traj_j_gaze)
        end = time.time()
        print('time for 50 heatmaps: ', end-start)

    print('generated 10,000 gaze heatmp pais of size: ', heatmap_size)
    return human_gaze