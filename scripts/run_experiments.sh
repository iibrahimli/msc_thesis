#! /bin/bash

# given experiment directories like /conf/experiment/1, /conf/experiment/2, etc.
# run given experiments (for all YAML config files in experiment dir) one after the other

# check if the number of arguments is greater than 0
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <experiment1> <experiment2> ..."
    exit 1
fi

# get the experiment numbers from the command line arguments
experiments=$@

# for the experiment number dur in conf/experiment/, run all the YAML config files
for experiment in $experiments; do
    for config in "arithmetic_lm/conf/experiment/$experiment"/*; do
        config_file=$(basename $config)
        echo -e "\n====================="
        echo "Running: python -m arithmetic_lm.train +experiment=$experiment/$config_file"
        python -m arithmetic_lm.train +experiment=$experiment/$config_file
        echo -e "=====================\n" 
    done
done