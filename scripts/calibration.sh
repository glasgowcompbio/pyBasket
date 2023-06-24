#!/bin/bash

# Source the utility functions
source utils.sh

# Check if debug mode is used
check_debug_mode $@

# Print all parameters for the script, depending on whether debug is on or not
print_parameters

# No clustering info
python calibrate_simulate.py --calibrate --num_burn_in $num_burn_in --num_posterior_samples $num_posterior_samples --num_chains $num_chains --num_sim $num_sim --parallel --n_jobs $n_jobs
check_and_remove_files

# With clustering info, n_clusters = 5
python calibrate_simulate.py --calibrate --num_burn_in $num_burn_in --num_posterior_samples $num_posterior_samples --num_chains $num_chains --num_sim $num_sim --with_clustering_info --n_clusters 5 --parallel --n_jobs $n_jobs
check_and_remove_files

# With clustering info, n_clusters = 10
python calibrate_simulate.py --calibrate --num_burn_in $num_burn_in --num_posterior_samples $num_posterior_samples --num_chains $num_chains --num_sim $num_sim --with_clustering_info --n_clusters 10 --parallel --n_jobs $n_jobs
check_and_remove_files