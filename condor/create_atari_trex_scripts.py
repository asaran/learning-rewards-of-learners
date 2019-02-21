def create_script_file(env_name):
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
    
    script = '#!/usr/bin/env bash\n'
    script += 'source ~/.bashrc\n'
    script += 'conda activate deeplearning\n'
    script += 'cd ../learner/\n'
    script += "python LearnAtariHumanSnippetsSorted.py --env_name " + env_name + " --reward_model_path ./learned_models/" + env_name + "_human.params --data_dir /scratch/cluster/dsbrown/atari_v1/"
    print(script)
    f = open("trex_" + env_name,'w')
    f.write(script)
    f.close()
    

#envs = ['mspacman', 'videopinball', 'hero', 'beamrider', 'qbert', 'seaquest', 'breakout', 'spaceinvaders', 'pong', 'enduro' ]
envs = ['qbert', 'spaceinvaders', 'mspacman']

for e in envs:
    print("+"*20)
    print(e)
    print("+"*20)
    create_script_file(e)
