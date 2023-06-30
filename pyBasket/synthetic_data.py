import numpy as np
import pandas as pd
from scipy.special import expit as logistic
from scipy.stats import halfnorm

# Define number of patients, tissues, and clusters
GENERATE_PYBASKET_N_PATIENTS = 500
GENERATE_PYBASKET_N_TISSUES = 25
GENERATE_PYBASKET_N_CLUSTERS = 10


def generate_pyBasket_data(
        n_patients=GENERATE_PYBASKET_N_PATIENTS,
        n_tissues=GENERATE_PYBASKET_N_TISSUES,
        n_clusters=GENERATE_PYBASKET_N_CLUSTERS):

    # Generate unique tissue and cluster indices for each patient
    basket_coords = np.arange(n_tissues)  # Tissue indices
    cluster_coords = np.arange(n_clusters)  # Cluster indices

    # Randomly assign a tissue and a cluster to each patient
    basket_idx = np.random.choice(basket_coords, size=n_patients)
    cluster_idx = np.random.choice(cluster_coords, size=n_patients)

    # Generate synthetic responsiveness data for each tissue
    theta_basket = np.random.normal(loc=0, scale=2, size=n_tissues)

    # Generate unique prior mean and std for each cluster in theta_cluster
    # These means represent the baseline effect for each cluster
    prior_means = np.random.normal(loc=0, scale=2, size=n_clusters)
    prior_std_mean = 0  # mean of the half-normal distribution
    prior_std_std = 1  # standard deviation of the half-normal distribution

    # Generate a set of standard deviations from a half-normal distribution
    # These are used to generate normally distributed data for the clusters
    prior_stds = halfnorm.rvs(loc=prior_std_mean, scale=prior_std_std, size=n_clusters)

    # Generate normally distributed data for each cluster, each with its own mean
    # and standard deviation
    theta_cluster = np.zeros((n_tissues, n_clusters))
    for i in range(n_clusters):
        theta_cluster[:, i] = np.random.normal(loc=prior_means[i], scale=prior_stds[i],
                                               size=n_tissues)

    # Convert the generated theta values to probabilities using the logistic function
    true_basket_p = logistic(theta_basket)
    true_cluster_p = logistic(theta_cluster)

    # Reshape the basket probabilities and compute the product of basket and cluster probabilities
    true_basket_reshaped = true_basket_p.reshape((n_tissues, 1))
    true_mat = true_basket_reshaped * true_cluster_p

    # Determine the true probability for each patient by indexing into the probability matrix
    # with the assigned tissue and cluster
    true_patient_p = true_mat[basket_idx, cluster_idx]

    # Draw a sample for each patient from a Bernoulli distribution with the patient's
    # true probability
    is_responsive = np.random.binomial(n=1, p=true_patient_p)

    # Create synthetic data dataframe containing the tissue and cluster assignments and
    # responsiveness for each patient
    data_df = pd.DataFrame({
        'basket_number': basket_idx,
        'cluster_number': cluster_idx,
        'responsive': is_responsive
    })

    results = {
        'data_df': data_df,
        'true_basket_p': true_basket_p,
        'true_cluster_p': true_cluster_p,
        'true_mat': true_mat,
        'n_patients': n_patients,
        'n_tissues': n_tissues,
        'n_clusters': n_clusters
    }
    return results