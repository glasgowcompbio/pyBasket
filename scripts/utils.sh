#!/bin/bash

check_debug_mode() {
  # Declare global DEBUG flag
  DEBUG=false

  # Check for '--debug' flag
  for arg in "$@"
  do
      if [ "$arg" = "--debug" ] ; then
          DEBUG=true
      fi
  done

  # Set parameters based on the DEBUG flag
  if [ "$DEBUG" = true ] ; then
      num_burn_in=100
      num_posterior_samples=100
      num_chains=1
      num_sim=5
      n_jobs=5
      echo -e "\n\033[1;34m--- DEBUG MODE ENABLED ---\033[0m"
  else
      num_burn_in=5000
      num_posterior_samples=5000
      num_chains=1
      num_sim=500
      n_jobs=50
      echo -e "\n\033[1;32m--- NORMAL MODE ---\033[0m"
  fi
}

print_parameters() {
  echo -e "Using the following parameters:"
  echo -e "\033[1;33mnum_burn_in:\033[0m" $num_burn_in
  echo -e "\033[1;33mnum_posterior_samples:\033[0m" $num_posterior_samples
  echo -e "\033[1;33mnum_chains:\033[0m" $num_chains
  echo -e "\033[1;33mnum_sim:\033[0m" $num_sim
  echo -e "\033[1;33mn_jobs:\033[0m" $n_jobs
  echo -e "\n"
}

check_and_remove_files() {
  # Check if hostname is 'cauchy'
  if [ "$(hostname)" = "cauchy" ]; then
      rm -rf /home/joewandy/.pytensor/compiledir_Linux-5.4--generic-x86_64-with-glibc2.27-x86_64-3.9.0-64/
  fi
}
