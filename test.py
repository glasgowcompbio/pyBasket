import sys
from os.path import exists

sys.path.append('..')
sys.path.append('.')

import numpy as np
import pandas as pd
import arviz as az

from pyBasket.env import Trial, TrueResponseSiteWithFeatures
from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF

DEBUG = True

num_burn_in = 1E5
num_posterior_samples = 1E5
num_chains = None # let pymc decide

if DEBUG:
    num_burn_in = 5E4
    num_posterior_samples = 5E4
    num_chains = 1

K = 6    # the number of groups
p0 = 0.20 # null response rate
p1 = 0.40 # target response rate

true_response_rates = [p0, p0, p0, p1, p1, p1]
enrollment = [14, 10]

n = 100
pvals_map = {
    p0: [1/10] * 10,
    p1: ([0.05] * 5) + ([0.15] * 5)
}

sites = []
for k in range(K):
    true_response_rate = true_response_rates[k]
    pvals = pvals_map[true_response_rate]
    site = TrueResponseSiteWithFeatures(k, true_response_rate, enrollment, n, pvals)
    sites.append(site)

evaluate_interim = [True, True] # evaluate every interim stage
analysis_names = ['independent', 'hierarchical', 'bhm']

futility_cutoff = DEFAULT_FUTILITY_CUTOFF
efficacy_cutoff = DEFAULT_EFFICACY_CUTOFF
early_futility_stop = True
early_efficacy_stop = False

trial = Trial(K, p0, p1, sites, evaluate_interim,
              num_burn_in, num_posterior_samples, analysis_names,
              futility_cutoff=futility_cutoff, efficacy_cutoff=efficacy_cutoff,
              early_futility_stop=early_futility_stop,
              early_efficacy_stop=early_efficacy_stop,
              num_chains=num_chains)

done = trial.reset()
while not done:
    done = trial.step()