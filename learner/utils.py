import numpy as np
import cv2
import csv
import os
import torch

def get_atari_head_demos(env_name, data_dir):
    print('env name: ', env_name)
    # read the meta data csv file
    trial_nums = []
    with open(data_dir+'meta_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                game_name = row[0].lower()
                # print(game_name)
                if game_name==env_name:
                    # print(game_name)
                    trial_nums.append(row[1])
                line_count += 1
        #print(f'Processed {line_count} lines.')
        # print(trial_nums)
    # enumerate folder names for the correct game (env_name)
    d = data_dir+'/'+env_name
    trials = [o for o in os.listdir(d) 
                        if os.path.isdir(os.path.join(d,o))]
    # print(trials)
    valid_trials = [t for t in trials if t.split('_')[0] in trial_nums]

    print('valid trials:', valid_trials)
    # accumulate stacks of 4 frames along the trajectory with an associated return of the 4th frame
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
