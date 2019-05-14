
env_names = [['breakout','Breakout'],['hero','Hero'], ['seaquest','Seaquest'], ['spaceinvaders','SpaceInvaders'], ['mspacman','MsPacman']]
snippet_lengths = ['50','250','500']
gaze_reg = ['0.001', '0.01', '0.1', '0.5', '1.0']
exp_type = ['rewardLearn','PPO','eval']
traj_sort_type = ['rewards','returns']
mask_scores = [['True','mask'], ['False','no-mask']]
use_gaze = [['True','gaze'],['False','no-gaze']]
gaze_loss_type = {'gaze':['coverage','EMD'],'no-gaze':['','']}

for exp in exp_type:
  for l in snippet_lengths:
    #bash_file_name = exp+'_'+l+'.sh'
    #f = open(bash_file_name,'w')
    for env in env_names:		
      #screen_name = exp+'_'+l+'_'+env[0]
      for g in gaze_reg:
        for t in traj_sort_type:
          for m in mask_scores:
            for ug in use_gaze:
                for gl in gaze_loss_type[ug[1]]:
                  bash_file_name = exp+'_'+l+"_"+ug[1]+"_"+gl+"_"+m[1]+"_"+t+"_"+g+'.sh'
                  f = open(bash_file_name,'w')

                  screen_name = exp+'_'+l+'_'+env[0]
                  f.write("#!/bin/bash")
                  f.write("screen -dmS "+screen_name+" bash")
                  f.write("screen -S "+screen_name+" -X stuff \"source ~/.virtualenv/gym/bin/activate")
                  f.write("\"")
                  if(exp_type=='rewardLearn'):
                    f.write("screen -S "+screen_name+" -X stuff \"CUDA_VISIBLE_DEVICES=0 python LearnAtariGazeHumanTrajs.py --env_name "+env[0]+" --data_dir ../data/atari-head/ --reward_model_path learned_models/"+env[0]+"_"+l+"_"+ug[1]+"_"+gl+"_"+m[1]+"_"+t+"_"+g+" --snippet_len "+l+" --use_gaze "+ug[0]+" --gaze_loss "+gl+" --gaze_reg "+g+" --traj_sort "+t+" --mask_scores "+m[0])
    
                  elif(exp_type=='eval'):
                    f.write("screen -S "+screen_name+" -X stuff \"python evaluateLearnedPolicy_condor.py --env_name "+env[0]+" --checkpoint "+env[0]+"_"+l+"_"+ug[1]+"_"+gl+"_"+m[1]+"_"+t+"_"+g+"/checkpoints/03900")
                  elif exp_type=='PPO':
                    f.write("screen -S "+screen_name+" -X stuff \"CUDA_VISIBLE_DEVICES=0 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=path_to_logs/"+env[0]+"_"+l+"_"+ug[1]+"_"+gl+"_"+m[1]+"_"+t+"_"+g+" python -m baselines.run --alg=ppo2 --env="+env[1]+"NoFrameskip-v4 --save_interval=50 --custom_reward pytorch --custom_reward_path learned_models/"+env[0]+"_"+l+"_"+ug[1]+"_"+gl+"_"+m[1]+"_"+t+"_"+g+"/ --num_timesteps=2e7")
        
                  f.write("\"")
      
    f.close()
        
