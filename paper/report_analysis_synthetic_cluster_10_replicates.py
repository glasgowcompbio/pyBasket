import os
import sys

sys.path.append('..')
sys.path.append('.')

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_squared_error
from scipy.stats import halfnorm
import math
import seaborn as sns
import pylab as plt

from scipy.special import expit as logistic

from pyBasket.preprocessing import get_pivot_count_df
from pyBasket.model import get_model_pyBasket_nc
from pyBasket.model import get_model_simple, get_model_bhm_nc
from pyBasket.common import create_if_not_exist


def generate_data():
    # Define number of patients, tissues, and clusters
    n_patients = 500
    n_tissues = 25
    n_clusters = 10

    # Generate tissue and cluster indices for each patient
    basket_coords = np.arange(n_tissues)
    cluster_coords = np.arange(n_clusters)
    basket_idx = np.random.choice(basket_coords, size=n_patients)
    cluster_idx = np.random.choice(cluster_coords, size=n_patients)

    # Generate synthetic responsiveness data
    theta_basket = np.random.normal(loc=0, scale=2, size=n_tissues)

    # Generate unique prior mean and std for each column in theta_cluster
    prior_means = np.random.normal(loc=0, scale=2, size=n_clusters)
    prior_std_mean = 0  # mean of the half-normal distribution
    prior_std_std = 1  # standard deviation of the half-normal distribution
    prior_std_scale = np.sqrt(2) * prior_std_std / np.pi
    prior_stds = halfnorm.rvs(loc=prior_std_mean, scale=prior_std_scale, size=n_clusters)

    theta_cluster = np.zeros((n_tissues, n_clusters))
    for i in range(n_clusters):
        theta_cluster[:, i] = np.random.normal(loc=prior_means[i], scale=prior_stds[i],
                                               size=n_tissues)

    true_basket_p = logistic(theta_basket)
    true_cluster_p = logistic(theta_cluster)
    true_basket_reshaped = true_basket_p.reshape((n_tissues, 1))
    true_mat = true_basket_reshaped * true_cluster_p

    true_patient_p = true_mat[basket_idx, cluster_idx]
    is_responsive = np.random.binomial(n=1, p=true_patient_p)

    # Create synthetic data dataframe
    data_df = pd.DataFrame({
        'basket_number': basket_idx,
        'cluster_number': cluster_idx,
        'responsive': is_responsive
    })

    # Print the first few rows of the data dataframe
    return data_df, true_basket_p, true_cluster_p, n_patients, n_tissues, n_clusters


def run_experiment(data_df, true_basket_p, true_cluster_p,
                   n_tissues, n_clusters,
                   n_burn_in=int(5E3), n_sample=int(5E3),
                   target_accept=0.99):
    df_pivot = get_pivot_count_df(data_df)

    # Simple model
    model_s = get_model_simple(df_pivot, n_tissues, n_clusters)
    with model_s:
        trace_s = pm.sample(n_sample, tune=n_burn_in, idata_kwargs={'log_likelihood': True},
                            target_accept=target_accept)
    stacked_s = az.extract(trace_s)

    # BHM (Berry 2013)
    p0 = 0.2
    p1 = 0.4
    model_bhm = get_model_bhm_nc(df_pivot, p0, p1, n_tissues, n_clusters)
    with model_bhm:
        trace_h1 = pm.sample(n_sample, tune=n_burn_in, idata_kwargs={'log_likelihood': True},
                             target_accept=target_accept)
    stacked_h1 = az.extract(trace_h1)

    # pyBasket
    model_h2_nc = get_model_pyBasket_nc(data_df, n_tissues, n_clusters)
    with model_h2_nc:
        trace_h2 = pm.sample(n_sample, tune=n_burn_in, idata_kwargs={'log_likelihood': True},
                             target_accept=target_accept)
    stacked_h2 = az.extract(trace_h2)

    # calculate RMSE for basket probabilities
    actual = true_basket_p

    predicted_basket_s = np.mean(stacked_s.basket_p.values, axis=1)
    predicted_basket_h1 = np.mean(stacked_h1.basket_p.values, axis=1)
    predicted_basket_h2 = np.mean(stacked_h2.basket_p.values, axis=1)

    rmse_s = math.sqrt(mean_squared_error(actual, predicted_basket_s))
    rmse_h1 = math.sqrt(mean_squared_error(actual, predicted_basket_h1))
    rmse_h2 = math.sqrt(mean_squared_error(actual, predicted_basket_h2))

    rmse_basket_p = pd.DataFrame({
        'method': ['Simple', 'BHM', 'pyBasket'],
        'RMSE': [rmse_s, rmse_h1, rmse_h2]
    })

    # calculate RMSE for cluster probabilities
    actual = true_cluster_p

    predicted_cluster_h2 = np.mean(stacked_h2.cluster_p.values, axis=2)
    rmse_h2 = math.sqrt(mean_squared_error(actual, predicted_cluster_h2))

    rmse_cluster_p = pd.DataFrame({
        'method': ['pyBasket'],
        'RMSE': [rmse_h2]
    })

    return rmse_basket_p, rmse_cluster_p


repeat = 30
n_burn_in = int(5E3)
n_sample = int(5E3)

all_rmse_basket_p = []
all_rmse_cluster_p = []

for i in range(repeat):
    print(i)

    data_df, true_basket_p, true_cluster_p, n_patients, n_tissues, n_clusters = generate_data()
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
