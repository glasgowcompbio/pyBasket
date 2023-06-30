import numpy as np
import pandas as pd
from scipy.special import expit
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
    mu_basket = np.random.normal(loc=0, scale=2, size=n_tissues)
    sigma_basket = halfnorm.rvs(loc=0, scale=1, size=1)
    theta_basket = np.random.normal(loc=mu_basket, scale=sigma_basket, size=n_tissues)

    # Generate synthetic responsiveness data for each cluster
    mu_cluster = np.random.normal(loc=0, scale=2, size=n_clusters)
    sigma_cluster = halfnorm.rvs(loc=0, scale=1, size=n_clusters)
    theta_cluster = np.random.normal(loc=mu_cluster, scale=sigma_cluster, size=n_clusters)

    # Generate interaction effects between 'basket' and 'cluster'
    interaction_theta = np.random.normal(loc=0, scale=2, size=(n_tissues, n_clusters))

    # Convert the generated theta values to log-odds
    logit_basket = theta_basket
    logit_cluster = theta_cluster
    logit_interaction = interaction_theta

    # Calculate the log-odds for each patient as the sum of the 'basket', 'cluster', and interaction effects
    logit_p = []
    for i in range(n_patients):
        b = basket_idx[i]
        c = cluster_idx[i]
        logit_p.append(logit_basket[b] + logit_cluster[c] + logit_interaction[b, c])
    logit_p = np.array(logit_p)

    # Convert the log-odds to probabilities using the logistic function
    true_patient_p = expit(logit_p)

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

    true_basket_p = expit(logit_basket)
    true_cluster_p = expit(logit_cluster)
    true_interaction_p = expit(logit_interaction)

    # The true joint probability of baskets and clusters
    true_joint_p = (true_basket_p[:, None] + true_cluster_p) + true_interaction_p

    results = {
        'data_df': data_df,
        'true_basket_p': true_basket_p,
        'true_cluster_p': true_cluster_p,
        'true_interaction_p': true_interaction_p,
        'true_joint_p': true_joint_p,
        'n_patients': n_patients,
        'n_tissues': n_tissues,
        'n_clusters': n_clusters
    }
    return results
