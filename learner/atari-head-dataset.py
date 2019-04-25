from scipy import stats as st
import math
import csv
import os
from os import path, listdir
import numpy as np

class AtariHeadDataset():

    # TRAJS_SUBDIR = 'trajectories'
    # SCREENS_SUBDIR = 'screens'

    def __init__(self, env_name, data_path):
        
        '''
            Loads the dataset trajectories into memory. 
            data_path is the root of the dataset (the folder, which contains
            the 'screens' and 'trajectories' folders. 
        '''

        self.trajs_path = data_path  
        self.env_name = env_name    
        # self.screens_path = path.join(data_path, AtariDataset.SCREENS_SUBDIR)
    
        #check that the we have the trajs where expected
        assert path.exists(self.trajs_path)
        
        self.trajectories = self.load_trajectories()

        # compute the stats after loading
        self.stats = {}
        for g in self.trajectories.keys():
            self.stats[g] = {}
            nb_games = self.trajectories[g].keys()

            # TODO: separate episode wise

            total_frames = sum([len(self.trajectories[g][traj]) for traj in self.trajectories[g]])
            final_scores = [self.trajectories[g][traj][-1]['score'] for traj in self.trajectories[g]]

            self.stats[g]['total_replays'] = len(nb_games)
            self.stats[g]['total_frames'] = total_frames
            self.stats[g]['max_score'] = np.max(final_scores)
            self.stats[g]['min_score'] = np.min(final_scores)
            self.stats[g]['avg_score'] = np.mean(final_scores)
            self.stats[g]['stddev'] = np.std(final_scores)
            self.stats[g]['sem'] = st.sem(final_scores)


    def load_trajectories(self):

        print('env name: ', self.env_name)
        # read the meta data csv file
        trial_nums = []
        with open(self.trajs_path+'meta_data.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    game_name = row[0].lower()
                    if game_name==self.env_name:
                        trial_nums.append(row[1])
                    line_count += 1

        d = path.join(self.trajs_path, self.env_name)
        trials = [o for o in listdir(d) 
                            if path.isdir(path.join(d,o))]

        # discard trial numbers <180 (episode # not recorded)
        trial_nums = [t for t in trial_nums if t>=180]

        # trajectory folder names for the chosen env
        valid_trials = [t for t in trials if t.split('_')[0] in trial_nums]
        print('valid trials:', valid_trials)


        trajectories = {}
        extra_episodes = 0
        # for game in listdir(self.trajs_path):
        game = self.env_name
        trajectories[game] = {}
        # game_dir = path.join(self.trajs_path, game)
        game_dir = d
        for traj in listdir(game_dir):
            if(traj in valid_trials):
                curr_traj = []
                last_episode = 0
                with open(path.join(game_dir, traj+'.txt')) as f:
                    for i,line in enumerate(f):
                        #first line is the metadata, second is the header
                        if i > 1:
                            #TODO will fix the spacing and True/False/integer in the next replay session
                            #frame,reward,score,terminal, action
                            curr_data = line.rstrip('\n').split(',')

                            # curr_data = line.rstrip('\n').replace(" ","").split(',')
                            curr_trans = {}
                            curr_trans['frame']    = int(curr_data[0])
                            curr_trans['episode']  = int(curr_data[1])                           
                            curr_trans['score']    = int(curr_data[2])
                            curr_trans['duration']   = int(curr_data[3])
                            curr_trans['reward']   = int(curr_data[4])
                            curr_trans['action']   = int(curr_data[5])
                            curr_trans['gaze_positions']   = int(curr_data[6:])

                            # start a new current trajectory if next epiosde begins
                            # save traj number beginning from 0 for these initial episodes
                            if(curr_trans['episode']!=last_episode):
                                trajectories[game][extra_episodes] = curr_traj
                                curr_traj = []
                                extra_episodes += 1
                            else:
                                curr_traj.append(curr_trans)
                            last_episode = curr_trans['episode'] 
                trajectories[game][int(traj.split('_')[0])] = curr_traj
        return trajectories