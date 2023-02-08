import pymc as pm
from sklearn.preprocessing import StandardScaler


def get_model_simple(data_df):
    K = data_df.shape[0]
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    coords = {'basket': data_df.index}
    with pm.Model(coords=coords) as model:
        α = pm.Gamma('alpha', alpha=2, beta=0.5)
        β = pm.Gamma('beta', alpha=2, beta=0.5)
        θ = pm.Beta('theta', alpha=α, beta=β, dims='basket')
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
        θ = pm.Deterministic('theta', pm.math.invlogit(p), dims='basket')

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
        θ = pm.Deterministic('theta', pm.math.invlogit(p), dims='basket')

        # Define the likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')
        return model


def get_model_logres(data_df):
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values
    df = data_df.drop(['n_trial', 'n_success'], axis=1)

    coords = {'basket': df.index, 'cluster': df.columns}
    with pm.Model(coords=coords) as model:
        s_k = pm.ConstantData('data', df, dims=('basket', 'cluster'))

        # hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=10)
        σ_α = pm.HalfNormal('sigma_alpha', sigma=10)
        σ_β = pm.HalfNormal('sigma_beta', sigma=10, dims='cluster')

        # priors
        α = pm.Normal('alpha', mu=μ_α, sigma=σ_α, dims='basket')
        β = pm.Normal('beta', mu=0, sigma=σ_β, dims='cluster')

        # linear model
        p = pm.math.dot(s_k, β) + α
        θ = pm.Deterministic('theta', pm.math.invlogit(p), dims='basket')

        # likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')
        return model


def get_model_logres_nc(data_df):
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values
    df = data_df.drop(['n_trial', 'n_success'], axis=1)

    coords = {'basket': df.index, 'cluster': df.columns}
    with pm.Model(coords=coords) as model:
        s_k = pm.ConstantData('data', df, dims=('basket', 'cluster'))

        # define the standard normal random variables
        z_α = pm.Normal('z_alpha', mu=0, sigma=1, dims='basket')
        z_β = pm.Normal('z_beta', mu=0, sigma=1, dims='cluster')

        # Define hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=10)
        σ_α = pm.HalfNormal('sigma_alpha', sigma=10)
        σ_β = pm.HalfNormal('sigma_beta', sigma=10, dims='cluster')

        # Define priors
        α = pm.Deterministic('alpha', μ_α + (z_α * σ_α), dims='basket')
        β = pm.Deterministic('beta', z_β * σ_β, dims='cluster')

        # Define the linear model using dot product
        p = pm.math.dot(s_k, β) + α
        θ = pm.Deterministic('theta', pm.math.invlogit(p), dims='basket')

        # Define the likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks, dims='basket')
        return model
