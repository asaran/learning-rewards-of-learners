#!/bin/bash

#screen -dmS "test screen" bash -c 'gym; python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_50_no-gaze/checkpoints/03900; exec sh'

screen -dmS new_screen bash
#screen -S new_screen -X stuff "cd learned_models
#"
screen -S new_screen -X stuff "source ~/.virtualenv/gym/bin/activate
"
screen -S new_screen -X stuff "python evaluateLearnedPolicy_condor.py --env_name seaquest --checkpoint seaquest_50_no-gaze/checkpoints/03900
"
screen -S new_screen -X stuff "python --version
"

