import pymc as pm


def get_model_simple(data_df):
    K = data_df.shape[0]
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    coords = {'basket': data_df.index}
    with pm.Model(coords=coords) as model:
        # α = pm.Gamma('alpha', alpha=1, beta=1)
        # β = pm.Gamma('beta', alpha=1, beta=1)
        α = pm.Gamma('alpha', alpha=2, beta=0.5)
        β = pm.Gamma('beta', alpha=2, beta=0.5)
        θ = pm.Beta('basket_p', alpha=α, beta=β, dims='basket')
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')
        return model


def get_model_bhm(data_df):
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values
    df = data_df.drop(['n_trial', 'n_success'], axis=1)

    coords = {'basket': df.index}
    with pm.Model(coords=coords) as model:
        # hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=10)
        σ_α = pm.InverseGamma('sigma_alpha', alpha=0.0005, beta=0.000005)

        # priors
        α = pm.Normal('alpha', mu=μ_α, sigma=σ_α, dims='basket')

        # linear model
        p = α
        θ = pm.Deterministic('basket_p', pm.math.invlogit(p), dims='basket')

        # likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')
        return model


def get_model_bhm_nc(data_df):
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values
    df = data_df.drop(['n_trial', 'n_success'], axis=1)

    coords = {'basket': df.index}
    with pm.Model(coords=coords) as model:
        # define the standard normal random variables
        z_α = pm.Normal('z_alpha', mu=0, sigma=1, dims='basket')

        # Define hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=10)
        σ_α = pm.InverseGamma('sigma_alpha', alpha=0.0005, beta=0.000005)

        # Define priors
        α = pm.Deterministic('alpha', μ_α + (z_α * σ_α), dims='basket')

        # Define the linear model using dot product
        p = α
        θ = pm.Deterministic('basket_p', pm.math.invlogit(p), dims='basket')

        # Define the likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')
        return model


def get_model_pyBasket(data_df):
    basket_coords = data_df['tissues'].unique() if 'tissues' in data_df.columns.values else \
        data_df['basket_number'].unique()
    cluster_coords = data_df['cluster_number'].unique()
    coords = {'basket': basket_coords, 'cluster': cluster_coords}

    with pm.Model(coords=coords) as model:
        # Define hyper-priors
        μ_basket = pm.Normal('basket_mu', mu=0, sigma=2, dims='basket')
        μ_cluster = pm.Normal('cluster_mu', mu=0, sigma=2, dims='cluster')
        σ_basket = pm.HalfNormal('basket_sigma', sigma=1)
        σ_cluster = pm.HalfNormal('cluster_sigma', sigma=1, dims='cluster')

        # Define priors
        basket_θ = pm.Normal('basket_theta', mu=μ_basket, sigma=σ_basket, dims='basket')
        cluster_θ = pm.Normal('cluster_theta', mu=μ_cluster, sigma=σ_cluster,
                              dims=('basket', 'cluster'))

        basket_p = pm.Deterministic('basket_p', pm.math.invlogit(basket_θ), dims='basket')
        cluster_p = pm.Deterministic('cluster_p', pm.math.invlogit(cluster_θ),
                                     dims=('basket', 'cluster'))

        # responsive is a product of each combination of basket and cluster probabilities
        basket_idx = data_df['basket_number'].values
        cluster_idx = data_df['cluster_number'].values
        is_responsive = data_df['responsive'].values

        p = basket_p[basket_idx] * cluster_p[basket_idx, cluster_idx]
        y = pm.Bernoulli('y', p=p, observed=is_responsive)
        return model


def get_model_pyBasket_nc(data_df):
    basket_coords = data_df['tissues'].unique() if 'tissues' in data_df.columns.values else \
        data_df['basket_number'].unique()
    cluster_coords = data_df['cluster_number'].unique()
    coords = {'basket': basket_coords, 'cluster': cluster_coords}

    with pm.Model(coords=coords) as model:
        z_basket = pm.Normal('z_basket', mu=0, sigma=1, dims='basket')
        z_cluster = pm.Normal('z_cluster', mu=0, sigma=1, dims=('basket', 'cluster'))

        # Define hyper-priors
        μ_basket = pm.Normal('basket_mu', mu=0, sigma=2, dims='basket')
        μ_cluster = pm.Normal('cluster_mu', mu=0, sigma=2, dims='cluster')
        σ_basket = pm.HalfNormal('basket_sigma', sigma=1)
        σ_cluster = pm.HalfNormal('cluster_sigma', sigma=1, dims='cluster')

        # Define priors
        basket_θ = μ_basket + (z_basket * σ_basket)
        cluster_θ = μ_cluster + (z_cluster * σ_cluster)

        basket_p = pm.Deterministic('basket_p', pm.math.invlogit(basket_θ), dims='basket')
        cluster_p = pm.Deterministic('cluster_p', pm.math.invlogit(cluster_θ),
                                     dims=('basket', 'cluster'))

        # responsive is a product of each combination of basket and cluster probabilities
        basket_idx = data_df['basket_number'].values
        cluster_idx = data_df['cluster_number'].values
        is_responsive = data_df['responsive'].values

        p = basket_p[basket_idx] * cluster_p[basket_idx, cluster_idx]
        y = pm.Bernoulli('y', p=p, observed=is_responsive)
        return model
