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
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                game_name = row[0].lower()
                # print(game_name)
                if game_name==env_name:
                    # print(game_name)
                    trial_nums.append(row[1])
                line_count += 1
        print(f'Processed {line_count} lines.')
        # print(trial_nums)
    # enumerate folder names for the correct game (env_name)
    d = data_dir
    trials = [o for o in os.listdir(d) 
                        if os.path.isdir(os.path.join(d,o))]
    # print(trials)
    valid_trials = [t for t in trials if t.split('_')[0] in trial_nums]

    print('valid trials:', valid_trials)
    # accumulate stacks of 4 frames along the trajectory with an associated return of the 4th frame
    trajectories = []
    returns = []
    for t in valid_trials:
        traj = []
        r = []
        img_dir = data_dir+'/'+t
        game_file = data_dir+'/'+t+'.txt'
        with open(game_file) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines] 
        img_paths = [os.path.join(img_dir, o) for o in os.listdir(img_dir)]

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
            r.append(float(line[4])) # unclipped reward of 4th frame
        trajectories.append(traj)
        returns.append(r)

    # sample random trajectories of length 50 and the associated returns for that length

    # return lists of associated partial trajectories and returns
    return trajectories, returns