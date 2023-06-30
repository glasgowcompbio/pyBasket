import os
import sys

sys.path.append('..')
sys.path.append('.')

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_squared_error
import math
import seaborn as sns
import pylab as plt

from pyBasket.preprocessing import get_pivot_count_df
from pyBasket.model import get_model_pyBasket_nc, get_model_simple_bern
from pyBasket.common import create_if_not_exist
from pyBasket.synthetic_data import generate_pyBasket_data


def run_experiment(data_df, true_basket_p, true_cluster_p,
                   n_tissues, n_clusters,
                   n_burn_in=int(5E3), n_sample=int(5E3),
                   target_accept=0.99):
    # df_pivot = get_pivot_count_df(data_df)

    # Simple model
    model_s = get_model_simple_bern(data_df, n_tissues)
    with model_s:
        trace_s = pm.sample(n_sample, tune=n_burn_in, idata_kwargs={'log_likelihood': True},
                            target_accept=target_accept)
    stacked_s = az.extract(trace_s)

    # Simple model + clustering
    model_sc = get_model_simple_bern(data_df, n_tissues, n_cluster=n_clusters)
    with model_sc:
        trace_sc = pm.sample(n_sample, tune=n_burn_in, idata_kwargs={'log_likelihood': True},
                             target_accept=target_accept)
    stacked_sc = az.extract(trace_sc)

    # pyBasket
    model_pyBasket_nc = get_model_pyBasket_nc(data_df, n_tissues, n_clusters)
    with model_pyBasket_nc:
        trace_pyBasket = pm.sample(n_sample, tune=n_burn_in, idata_kwargs={'log_likelihood': True},
                                   target_accept=target_accept)
    stacked_pyBasket = az.extract(trace_pyBasket)

    # calculate RMSE for basket probabilities
    actual = true_basket_p

    predicted_basket_s = np.mean(stacked_s.basket_p.values, axis=1)
    rmse_s = math.sqrt(mean_squared_error(actual, predicted_basket_s))

    predicted_basket_sc = np.mean(stacked_sc.basket_p.values, axis=1)
    rmse_sc = math.sqrt(mean_squared_error(actual, predicted_basket_sc))

    predicted_basket_pyBasket = np.mean(stacked_pyBasket.basket_p.values, axis=1)
    rmse_pyBasket = math.sqrt(mean_squared_error(actual, predicted_basket_pyBasket))

    rmse_basket_p = pd.DataFrame({
        'method': ['S', 'S+C', 'pyBasket'],
        'RMSE': [rmse_s, rmse_sc, rmse_pyBasket]
    })

    # calculate RMSE for cluster probabilities
    actual = true_cluster_p

    predicted_cluster_sc = np.mean(stacked_sc.cluster_p.values, axis=2)
    predicted_cluster_pyBasket = np.mean(stacked_pyBasket.cluster_p.values, axis=2)
    rmse_sc = math.sqrt(mean_squared_error(actual, predicted_cluster_sc))
    rmse_pyBasket = math.sqrt(mean_squared_error(actual, predicted_cluster_pyBasket))

    rmse_cluster_p = pd.DataFrame({
        'method': ['S+C', 'pyBasket'],
        'RMSE': [rmse_sc, rmse_pyBasket]
    })

    return rmse_basket_p, rmse_cluster_p


repeat = 30
n_burn_in = int(5E3)
n_sample = int(5E3)

all_rmse_basket_p = []
all_rmse_cluster_p = []

for i in range(repeat):
    print(i)

    results = generate_pyBasket_data()
    data_df = results['data_df']
    true_basket_p = results['true_basket_p']
    true_cluster_p = results['true_cluster_p']
    n_tissues = results['n_tissues']
    n_clusters = results['n_clusters']

    rmse_basket_p, rmse_cluster_p = run_experiment(data_df, true_basket_p, true_cluster_p,
                                                   n_tissues, n_clusters,
                                                   n_burn_in=n_burn_in, n_sample=n_sample)

    all_rmse_basket_p.append(rmse_basket_p)
    all_rmse_cluster_p.append(rmse_cluster_p)

assert len(all_rmse_basket_p) == len(all_rmse_cluster_p)

for i in range(len(all_rmse_basket_p)):
    basket_df = all_rmse_basket_p[i]
    cluster_df = all_rmse_cluster_p[i]
    basket_df['repeat'] = i
    cluster_df['repeat'] = i

all_rmse_basket_p_df = pd.concat(all_rmse_basket_p)
all_rmse_cluster_p_df = pd.concat(all_rmse_cluster_p)

out_dir = os.path.abspath('results')
create_if_not_exist(out_dir)

all_rmse_basket_p_df.to_pickle(os.path.join(out_dir, 'all_rmse_basket_p_df.p'))
all_rmse_cluster_p_df.to_pickle(os.path.join(out_dir, 'all_rmse_cluster_p_df.p'))

out_dir = os.path.abspath('results')

all_rmse_basket_p_df = pd.read_pickle(os.path.join(out_dir, 'all_rmse_basket_p_df.p'))
all_rmse_cluster_p_df = pd.read_pickle(os.path.join(out_dir, 'all_rmse_cluster_p_df.p'))

sns.set_context('poster')

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

sns.boxplot(x='method', y='RMSE', data=all_rmse_basket_p_df, ax=ax1)
ax1.set_title('Root Mean Squared Error (RMSE) \n of basket probabilities')
plt.xlabel(None)
plt.tight_layout()
plt.savefig('results/report_all_rmse_basket.png', dpi=300)

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

sns.boxplot(x='method', y='RMSE', data=all_rmse_cluster_p_df, ax=ax1)
ax1.set_title('Root Mean Squared Error (RMSE) \n of cluster probabilities')
plt.xlabel(None)
plt.tight_layout()
plt.savefig('results/report_all_rmse_cluster.png', dpi=300)
