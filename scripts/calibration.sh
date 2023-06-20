#!/bin/bash

# DEBUG flag
DEBUG=true

# Set parameters based on the DEBUG flag
if [ "$DEBUG" = true ] ; then
    num_burn_in=1000
    num_posterior_samples=1000
    num_chains=1
    num_sim=3
    n_jobs=1
    echo -e "\n\033[1;34m--- DEBUG MODE ENABLED ---\033[0m"
else
    num_burn_in=5000
    num_posterior_samples=5000
    num_chains=1
    num_sim=500
    n_jobs=50
    echo -e "\n\033[1;32m--- NORMAL MODE ---\033[0m"
fi

# Print the parameters
echo -e "Using the following parameters:"
echo -e "\033[1;33mnum_burn_in:\033[0m" $num_burn_in
echo -e "\033[1;33mnum_posterior_samples:\033[0m" $num_posterior_samples
echo -e "\033[1;33mnum_chains:\033[0m" $num_chains
echo -e "\033[1;33mnum_sim:\033[0m" $num_sim
echo -e "\n"

# No clustering info
python calibration.py --num_burn_in $num_burn_in --num_posterior_samples $num_posterior_samples --num_chains $num_chains --num_sim $num_sim --no-with_clustering_info --parallel --n_jobs $n_jobs

# With clustering info, n_clusters = 5
python calibration.py --num_burn_in $num_burn_in --num_posterior_samples $num_posterior_samples --num_chains $num_chains --num_sim $num_sim --with_clustering_info --n_clusters 5 --parallel --n_jobs $n_jobs

# With clustering info, n_clusters = 10
python calibration.py --num_burn_in $num_burn_in --num_posterior_samples $num_posterior_samples --num_chains $num_chains --num_sim $num_sim --with_clustering_info --n_clusters 10 --parallel --n_jobs $n_jobs
