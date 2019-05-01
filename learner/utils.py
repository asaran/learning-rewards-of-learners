import numpy as np
import cv2
import csv
import os
import torch
from os import path, listdir
cv2.ocl.setUseOpenCL(False)

def normalize_state(obs):
    return obs / 255.0

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

def MaxSkipAndWarpFrames(trajectory_dir):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    num_frames = len(listdir(trajectory_dir))
    skip=4

    sample_pic = np.random.choice(listdir(trajectory_dir))
    image_path = path.join(trajectory_dir, sample_pic)
    pic = cv2.imread(image_path)
    obs_buffer = np.zeros((2,)+pic.shape, dtype=np.uint8)
    max_frames = []
    for i in range(num_frames):
        #TODO: check that i should max before warping.
        b = trajectory_dir.split("_")
        img_name =  "_".join(b[1:3]) + "_" + str(i) + ".png"
        if i % skip == skip - 2:
            # print(path.join(trajectory_dir, img_name))
            obs = cv2.imread(path.join(trajectory_dir, img_name))
            # print(type(obs))
            obs_buffer[0] = obs
        if i % skip == skip - 1:
            # print(path.join(trajectory_dir, img_name))
            obs = cv2.imread(path.join(trajectory_dir, img_name))
            obs_buffer[1] = obs
            #warp max to 80x80 grayscale
            image = obs_buffer.max(axis=0)
            warped = GrayScaleWarpImage(image)
            max_frames.append(warped)
    return max_frames

def StackFrames(frames):
    import copy
    """stack every four frames to make an observation (84,84,4)"""
    stacked = []
    stacked_obs = np.zeros((84,84,4))
    for i in range(len(frames)):
        if i >= 3:
            stacked_obs[:,:,0] = frames[i-3]
            stacked_obs[:,:,1] = frames[i-2]
            stacked_obs[:,:,2] = frames[i-1]
            stacked_obs[:,:,3] = frames[i]
            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0))
    return stacked

def get_atari_head_demos(env_name, data_dir):
    print('env name: ', env_name)
    # read the meta data csv file
    trial_nums = []
    with open(data_dir+'meta_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                game_name = row[0].lower()
                if game_name==env_name:
                    trial_nums.append(row[1])
                line_count += 1

    d = data_dir+'/'+env_name
    trials = [o for o in os.listdir(d) 
                        if os.path.isdir(os.path.join(d,o))]

    # trajectory folder names for the chosen game
    valid_trials = [t for t in trials if t.split('_')[0] in trial_nums]

    print('valid trials:', valid_trials)
    # accumulate stacks of 4 frames along the trajectory with an associated return of the 4th frame
    # TODO: check for episode number, separate trajectories by episodes
    trajectories = []
    returns = []
    gaze_maps = []
    for t in valid_trials:
        traj = []
        r = []
        gaze = []
        img_dir = data_dir+'/'+env_name+'/'+t
        game_file = data_dir+'/'+env_name+'/'+t+'.txt'        
        lines = read_gaze_file(game_file)
        img_paths = [os.path.join(img_dir, o) for o in os.listdir(img_dir)]
        # print(img_paths)

        for p in range(3,len(img_paths)):
            line = lines[p].split(',')
            imgs = [cv2.imread(img_paths[p-i]) for i in range(3,-1,-1)]
            im_gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
            # print(im_gray[0].shape)
            im_gray = [cv2.resize(im,(84,84)) for im in im_gray]
            imgs_stacked = np.stack(im_gray,2)
            imgs_stacked = np.expand_dims(imgs_stacked, axis=0)
            # imgs_stacked = torch.FloatTensor(imgs_stacked)
            # imgs_stacked = imgs_stacked.unsqueeze(0)
            # print(imgs_stacked.shape)
            traj.append(imgs_stacked)

            if line[4]!='null':
                r.append(float(line[4])) # unclipped reward of 4th frame
            else:
                r.append(0)
            
            gaze_points = line[6:]
            gaze_map = generate_gaze_map(gaze_points, imgs[0].shape)
            gaze.append(gaze_map)

        trajectories.append(traj)
        returns.append(r)
        gaze_maps.append(gaze)

    # return lists of associated partial trajectories and returns
    return trajectories, returns, gaze_maps


def get_sorted_traj_indices(env_name, dataset):
    #need to pick out a subset of demonstrations based on desired performance
    #first let's sort the demos by performance, we can use the trajectory number to index into the demos so just
    #need to sort indices based on 'score'
    g = env_name
    #Note, I'm also going to try only keeping the full demonstrations that end in terminal
    traj_indices = []
    traj_scores = []
    traj_dirs = []
    traj_rewards = []
    for t in dataset.trajectories[g]:
        # if env_name == "revenge":
        #     traj_indices.append(t)
        #     traj_scores.append(dataset.trajectories[g][t][-1]['score'])

        # elif dataset.trajectories[g][t][-1]['terminal']:
        traj_indices.append(t)
        traj_scores.append(dataset.trajectories[g][t][-1]['score'])
        traj_dirs.append(dataset.trajectories[g][t][-1]['img_dir'])
        traj_rewards.append([dataset.trajectories[g][t][i]['reward'] for i in range(len(dataset.trajectories[g][t]))])

    sorted_traj_indices = [x for _, x in sorted(zip(traj_scores, traj_indices), key=lambda pair: pair[0])]
    sorted_traj_scores = sorted(traj_scores)
    sorted_traj_dirs = [x for _, x in sorted(zip(traj_scores, traj_dirs), key=lambda pair: pair[0])]
    sorted_traj_rewards = [x for _, x in sorted(zip(traj_scores, traj_rewards), key=lambda pair: pair[0])]

    #print(sorted_traj_scores)
    #print(len(sorted_traj_scores))
    print("Max human score", max(sorted_traj_scores))
    print("Min human score", min(sorted_traj_scores))

    #so how do we want to get demos? how many do we have if we remove duplicates?
    seen_scores = set()
    non_duplicates = []
    for i,s,d,r in zip(sorted_traj_indices, sorted_traj_scores, sorted_traj_dirs, sorted_traj_rewards):
        if s not in seen_scores:
            seen_scores.add(s)
            non_duplicates.append((i,s,d, r))
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
    demos = non_duplicates[start:num_demos*skip + start:skip]
    #print("(index, score) pairs:",demos)
    return demos


def get_preprocessed_trajectories(env_name, dataset, data_dir):
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
    for indx, score, img_dir, rew in demos:
        human_scores.append(score)
        human_rewards.append(rew)
        # traj_dir = path.join(data_dir, 'screens', env_name, str(indx))
        traj_dir = path.join(data_dir, env_name, img_dir)
        #print("generating traj from", traj_dir)
        maxed_traj = MaxSkipAndWarpFrames(traj_dir)
        stacked_traj = StackFrames(maxed_traj)
        demo_norm_mask = []
        #normalize values to be between 0 and 1 and have top part masked
        for ob in stacked_traj:
            # demo_norm_mask.append(mask_score(normalize_state(ob), crop_top))
            demo_norm_mask.append(normalize_state(ob))  # currently not cropping
        human_demos.append(demo_norm_mask)
    return human_demos, human_scores, human_rewards


def read_gaze_file(game_file):
    with open(game_file) as f:
            lines = f.readlines()
    lines = [x.strip() for x in lines] 
    return lines
        

def generate_gaze_map(gaze, img_shape):
    gaze_map = np.zeros((img_shape[0],img_shape[1]))
    for j in range(0,len(gaze),2):
        if('null' not in gaze[j]):
            x = float(gaze[j])
            y = float(gaze[j+1])
            # print(gaze_map.shape)
            # print(y,x)
            # TODO: place a gaussian blob at gaze point
            gaze_map[min(int(y),img_shape[0]-1),min(int(x),img_shape[1]-1)] = 1.0
        else:
            # print('no gaze')
            nogaze = 1

    gaze_map = cv2.resize(gaze_map,(7,7))
    # normalize gaze map 
    gaze_min = np.min(gaze_map)
    gaze_max = np.max(gaze_map)
    # print(gaze_map)
    # print('gaze max:', gaze_max)
    # print('gaze min:', gaze_min)
    # TODO: denominator zero fix
    gaze_map = (gaze_map-gaze_min)/(gaze_max-gaze_min)

    # threshold values for a binary map
    return gaze_map
