import pymc as pm
import numpy as np


def get_model_simple(data_df):
    '''
    Construct a probabilistic model using PyMC3.

    This function builds a hierarchical Bayesian model where each
    "basket" has its own success probability, but all the success
    probabilities are assumed to come from a common Beta distribution.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data frame where each row corresponds to a "basket".
        It must contain two columns:
            - 'n_trial': the number of trials for each basket.
            - 'n_success': the number of successes for each basket.

    Returns
    -------
    model : pm.Model
        The compiled PyMC3 model ready for fitting.

    '''
    # Total number of baskets
    K = data_df.shape[0]

    # Number of trials and successes for each basket
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    # Setting the 'basket' to data frame's index
    coords = {'basket': data_df.index}

    # Constructing the model
    with pm.Model(coords=coords) as model:
        # Hyperparameters of the Beta distribution are assumed to
        # follow a Gamma distribution.
        α = pm.Gamma('alpha', alpha=1, beta=1)  # Alpha parameter of the Beta distribution
        β = pm.Gamma('beta', alpha=1, beta=1)  # Beta parameter of the Beta distribution

        # Success probabilities for each basket, assumed to follow a Beta distribution
        θ = pm.Beta('basket_p', alpha=α, beta=β, dims='basket')

        # The observed successes for each basket are assumed to follow
        # a Binomial distribution with the basket-specific success probabilities.
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')

        # Return the model
        return model


def get_model_bhm(data_df, p0, p1):
    '''
    Construct a Bayesian Hierarchical Model using PyMC3.

    This function implements a hierarchical Bayesian model from Berry (2013)
    to infer basket response rates. Note unlike Berry, we don't implement the
    targeted rate adjustment here.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data frame where each row corresponds to a "basket".
        It must contain two columns:
            - 'n_trial': the number of trials for each basket.
            - 'n_success': the number of successes for each basket.

    Returns
    -------
    model : pm.Model
        The compiled PyMC3 model ready for fitting.

    '''
    # Number of trials and successes for each basket
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    # Remove 'n_trial' and 'n_success' columns from data frame
    df = data_df.drop(['n_trial', 'n_success'], axis=1)

    # Setting the 'basket' to data frame's index
    coords = {'basket': df.index}

    # mu0 = log(p0/(1-p0)) where p0 denotes the baseline or standard of care response rate
    # if p0 is 0.2, then mu0 is rougly -1.34, which is the same as Berry (2013).
    mu0 = np.log(p0/(1-p0))
    response_adj = np.log(p1/(1-p1))

    # Constructing the model
    with pm.Model(coords=coords) as model:
        # Hyper-priors for the mean and variance of the alpha parameters
        μ_α = pm.Normal('mu_alpha', mu=mu0, sigma=10)  # Mean of the alpha parameters
        σ_α = pm.InverseGamma('sigma_alpha', alpha=0.0005,
                              beta=0.000005)  # Variance of the alpha parameters

        # Prior for the alpha parameters of the logistic function
        α = pm.Normal('alpha', mu=μ_α, sigma=σ_α, dims='basket')

        # Logistic model for the success probabilities
        p = α
        p_adj = pm.Deterministic('p_adj', α - response_adj, dims='basket')
        θ = pm.Deterministic('basket_p', pm.math.invlogit(p), dims='basket')

        # The observed successes for each basket are assumed to follow
        # a Binomial distribution with the basket-specific success probabilities.
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')

        # Return the model
        return model


def get_model_bhm_nc(data_df, p0, p1):
    '''
    Construct a non-centered Bayesian Hierarchical Model using PyMC3.

    This function builds a non-centered hierarchical Bayesian model from Berry (2013)
    as implemented in get_model_bhm() above. In a non-centered parameterization,
    the model separates the mean and standard deviation of the group-level effects
    from the individual group effects, making the model more efficient and robust.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data frame where each row corresponds to a "basket".
        It must contain two columns:
            - 'n_trial': the number of trials for each basket.
            - 'n_success': the number of successes for each basket.

    Returns
    -------
    model : pm.Model
        The compiled PyMC3 model ready for fitting.

    '''
    # Number of trials and successes for each basket
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    # Remove 'n_trial' and 'n_success' columns from data frame
    df = data_df.drop(['n_trial', 'n_success'], axis=1)

    # Setting the 'basket' to data frame's index
    coords = {'basket': df.index}

    # mu0 = log(p0/(1-p0)) where p0 denotes the baseline or standard of care response rate
    # if p0 is 0.2, then mu0 is rougly -1.34, which is the same as Berry (2013).
    mu0 = np.log(p0/(1-p0))
    response_adj = np.log(p1/(1-p1))

    # Constructing the model
    with pm.Model(coords=coords) as model:
        # Define the standard normal random variables for non-centered parameterization
        z_α = pm.Normal('z_alpha', mu=0, sigma=1, dims='basket')

        # Define hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=10)  # Mean of the alpha parameters
        σ_α = pm.InverseGamma('sigma_alpha', alpha=0.0005, beta=0.000005)  # Variance of the alpha parameters

        # Define priors using non-centered parameterization
        α = pm.Deterministic('alpha', μ_α + (z_α * σ_α), dims='basket')

        # Define the linear model using dot product
        p = α
        p_adj = pm.Deterministic('p_adj', α - response_adj, dims='basket')
        θ = pm.Deterministic('basket_p', pm.math.invlogit(p), dims='basket')

        # Define the likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')

        # Return the model
        return model



def get_model_simple_bern(data_df):
    '''
    Construct a probabilistic model using PyMC3 to assess patient response.

    This function builds a Bayesian model where each "basket" (e.g., tissue type or
    other patient subgroup) and "cluster" (e.g., treatment group) has its own success
    probability. The probabilities are then combined to calculate the response rate
    for each combination of basket and cluster.

    Note that unlike get_model_simple(), here we use Bernoulli likelihood rather than Binomial
    for the observations.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data frame where each row corresponds to a patient. It must contain
        the following columns:
            - 'tissues' or 'basket_number': the group identifier for each patient.
            - 'cluster_number': the cluster identifier for each patient.
            - 'responsive': a binary variable indicating whether the patient responded to treatment.

    Returns
    -------
    model : pm.Model
        The compiled PyMC3 model ready for fitting.

    '''
    # Unique identifiers for each basket and cluster
    basket_coords = data_df['tissues'].unique() if 'tissues' in data_df.columns.values else \
        data_df['basket_number'].unique()
    cluster_coords = data_df['cluster_number'].unique()

    # Setting the 'basket' and 'cluster' coordinates
    coords = {'basket': basket_coords, 'cluster': cluster_coords}

    # Constructing the model
    with pm.Model(coords=coords) as model:
        # Prior probability of each basket being responsive
        basket_p = pm.Beta('basket_p', alpha=1, beta=1, dims='basket')

        # Prior probability of each cluster being responsive for each basket
        cluster_p = pm.Beta('cluster_p', alpha=1, beta=1, dims=('basket', 'cluster'))

        # The response variable is a product of each combination of
        # basket and cluster probabilities
        basket_idx = data_df['basket_number'].values
        cluster_idx = data_df['cluster_number'].values
        is_responsive = data_df['responsive'].values
        y = pm.Bernoulli('y', p=basket_p[basket_idx] * cluster_p[basket_idx, cluster_idx],
                         observed=is_responsive)

        # Return the model
        return model



def get_model_hierarchical_bern(data_df):
    '''
    Construct a hierarchical probabilistic model using PyMC3 to assess patient response.

    This function extends get_patient_model_simple() by adding some hierarchies
    on the "basket" (e.g., tissue type or other patient subgroup) and "cluster" (from omics data)
    has its own success probability. The probabilities are then combined to calculate
    the response rate for each combination of basket and cluster.

    The model suffers from divergence problem during sampling, but it's difficult to fix it
    since there is no conjugate prior to the Beta distributions. See:

    - https://discourse.pymc.io/t/non-centered-parameterization-of-a-beta-distribution/6872/7
    - https://stats.stackexchange.com/questions/67443/does-the-beta-distribution-have-a-conjugate-prior

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data frame where each row corresponds to a patient. It must contain
        the following columns:
            - 'tissues' or 'basket_number': the group identifier for each patient.
            - 'cluster_number': the treatment identifier for each patient.
            - 'responsive': a binary variable indicating whether the patient responded to treatment.

    Returns
    -------
    model : pm.Model
        The compiled PyMC3 model ready for fitting.

    '''
    # Unique identifiers for each basket and cluster
    basket_coords = data_df['tissues'].unique() if 'tissues' in data_df.columns.values else \
        data_df['basket_number'].unique()
    cluster_coords = data_df['cluster_number'].unique()

    # Setting the 'basket' and 'cluster' coordinates
    coords = {'basket': basket_coords, 'cluster': cluster_coords}

    # Constructing the model
    with pm.Model(coords=coords) as model:
        # Hyperpriors for the parameters of the Beta distribution for basket_p and cluster_p
        basket_alpha = pm.Beta('basket_alpha', alpha=1, beta=1)
        basket_beta = pm.Beta('basket_beta', alpha=1, beta=1)
        cluster_alpha = pm.Beta('cluster_alpha', alpha=1, beta=1, dims='cluster')
        cluster_beta = pm.Beta('cluster_beta', alpha=1, beta=1, dims='cluster')

        # Prior probability of each basket being responsive
        basket_p = pm.Beta('basket_p', alpha=basket_alpha, beta=basket_beta, dims='basket')

        # Prior probability of each cluster being responsive for each basket
        cluster_p = pm.Beta('cluster_p', alpha=cluster_alpha, beta=cluster_beta,
                            dims=('basket', 'cluster'))

        # The response variable is a product of each combination of
        # basket and cluster probabilities
        basket_idx = data_df['basket_number'].values
        cluster_idx = data_df['cluster_number'].values
        is_responsive = data_df['responsive'].values
        y = pm.Bernoulli('y', p=basket_p[basket_idx] * cluster_p[basket_idx, cluster_idx],
                         observed=is_responsive)

        # Return the model
        return model



def get_model_pyBasket(data_df):
    '''
    Construct a probabilistic model using PyMC3 for patient response,
    considering different tissues and clusters with different success probabilities.

    This function builds a Bayesian model where each "basket" (e.g., tissue type or
    other patient subgroup) and "cluster" (from omics data) has its own success
    probability following a logistic-normal distribution. These probabilities are then
    combined to calculate the response rate for each combination of basket and cluster.

    Unlike get_patient_model_hierarchical() above, here we use Normal distributions to model
    the log of success probabilities, so this is more amenable to efficient sampling.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data frame where each row corresponds to a patient. It must contain
        the following columns:
            - 'tissues' or 'basket_number': the group identifier for each patient.
            - 'cluster_number': the treatment identifier for each patient.
            - 'responsive': a binary variable indicating whether the patient responded to treatment.

    Returns
    -------
    model : pm.Model
        The compiled PyMC3 model ready for fitting.

    '''
    # Unique identifiers for each basket and cluster
    basket_coords = data_df['tissues'].unique() if 'tissues' in data_df.columns.values else \
        data_df['basket_number'].unique()
    cluster_coords = data_df['cluster_number'].unique()

    # Setting the 'basket' and 'cluster' coordinates
    coords = {'basket': basket_coords, 'cluster': cluster_coords}

    # Constructing the model
    with pm.Model(coords=coords) as model:
        # Define hyper-priors
        μ_basket = pm.Normal('basket_mu', mu=0, sigma=2, dims='basket')
        μ_cluster = pm.Normal('cluster_mu', mu=0, sigma=2, dims='cluster')
        σ_basket = pm.HalfNormal('basket_sigma', sigma=1)
        σ_cluster = pm.HalfNormal('cluster_sigma', sigma=1, dims='cluster')

        # Define priors using the logistic-normal distribution for each basket and cluster
        basket_θ = pm.Normal('basket_theta', mu=μ_basket, sigma=σ_basket, dims='basket')
        cluster_θ = pm.Normal('cluster_theta', mu=μ_cluster, sigma=σ_cluster,
                              dims=('basket', 'cluster'))

        # Calculate the success probabilities using the logistic function
        basket_p = pm.Deterministic('basket_p', pm.math.invlogit(basket_θ), dims='basket')
        cluster_p = pm.Deterministic('cluster_p', pm.math.invlogit(cluster_θ),
                                     dims=('basket', 'cluster'))

        # The response variable is a product of each combination of
        # basket and cluster probabilities
        basket_idx = data_df['basket_number'].values
        cluster_idx = data_df['cluster_number'].values
        is_responsive = data_df['responsive'].values

        p = basket_p[basket_idx] * cluster_p[basket_idx, cluster_idx]
        y = pm.Bernoulli('y', p=p, observed=is_responsive)

        # Return the model
        return model



def get_model_pyBasket_nc(data_df):
    '''
    Constructs a probabilistic model using PyMC3 for patient response,
    considering different tissues and clusters with different success probabilities.
    This version of the pyBasket model introduces non-centered parametrization.

    Non-centered parametrization helps to improve sampling efficiency and can
    mitigate issues like divergences in Hamiltonian Monte Carlo (HMC) sampling.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data frame where each row corresponds to a patient. It must contain
        the following columns:
            - 'tissues' or 'basket_number': the group identifier for each patient.
            - 'cluster_number': the treatment identifier for each patient.
            - 'responsive': a binary variable indicating whether the patient responded to treatment.

    Returns
    -------
    model : pm.Model
        The compiled PyMC3 model ready for fitting.

    '''
    # Unique identifiers for each basket and cluster
    basket_coords = data_df['tissues'].unique() if 'tissues' in data_df.columns.values else \
        data_df['basket_number'].unique()
    cluster_coords = data_df['cluster_number'].unique()

    # Setting the 'basket' and 'cluster' coordinates
    coords = {'basket': basket_coords, 'cluster': cluster_coords}

    # Constructing the model
    with pm.Model(coords=coords) as model:
        # Define standard normal random variables for non-centered parametrization
        z_basket = pm.Normal('z_basket', mu=0, sigma=1, dims='basket')
        z_cluster = pm.Normal('z_cluster', mu=0, sigma=1, dims=('basket', 'cluster'))

        # Define hyper-priors
        μ_basket = pm.Normal('basket_mu', mu=0, sigma=2, dims='basket')
        μ_cluster = pm.Normal('cluster_mu', mu=0, sigma=2, dims='cluster')
        σ_basket = pm.HalfNormal('basket_sigma', sigma=1)
        σ_cluster = pm.HalfNormal('cluster_sigma', sigma=1, dims='cluster')

        # Define priors, using non-centered parametrization
        basket_θ = μ_basket + (z_basket * σ_basket)
        cluster_θ = μ_cluster + (z_cluster * σ_cluster)

        # Calculate the success probabilities using the logistic function
        basket_p = pm.Deterministic('basket_p', pm.math.invlogit(basket_θ), dims='basket')
        cluster_p = pm.Deterministic('cluster_p', pm.math.invlogit(cluster_θ),
                                     dims=('basket', 'cluster'))

        # The response variable is a product of each combination of
        # basket and cluster probabilities
        basket_idx = data_df['basket_number'].values
        cluster_idx = data_df['cluster_number'].values
        is_responsive = data_df['responsive'].values

        p = basket_p[basket_idx] * cluster_p[basket_idx, cluster_idx]
        y = pm.Bernoulli('y', p=p, observed=is_responsive)

        # Return the model
        return model
