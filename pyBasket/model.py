import pymc as pm
from sklearn.preprocessing import StandardScaler


def get_model_simple(data_df):
    K = data_df.shape[0]
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values
    with pm.Model() as model:
        α = pm.Gamma('alpha', alpha=2, beta=0.5)
        β = pm.Gamma('beta', alpha=2, beta=0.5)
        θ = pm.Beta('theta', alpha=α, beta=β, shape=K)
        y = pm.Binomial('y', n=ns, p=θ, observed=ks)
        return model


def get_model_bhm(data_df):
    K = data_df.shape[0]
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    with pm.Model() as model:
        # hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=3)
        σ_α = pm.HalfNormal('sigma_alpha', sigma=3)

        # priors
        α = pm.Normal('alpha', mu=μ_α, sigma=σ_α, shape=K)
        p = α
        θ = pm.Deterministic('theta', pm.math.invlogit(p))

        # likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks)
        return model


def get_model_bhm_nc(data_df):
    K = data_df.shape[0]
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    with pm.Model() as model:
        # define the standard normal random variables
        z_α = pm.Normal('z_alpha', mu=0, sigma=1, shape=K)

        # Define hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=3)
        σ_α = pm.HalfNormal('sigma_alpha', sigma=3)

        # Define priors
        α = pm.Deterministic('alpha', μ_α + (z_α * σ_α))

        # Define the linear model using dot product
        p = α
        θ = pm.Deterministic('theta', pm.math.invlogit(p))

        # Define the likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks)
        return model


def get_model_logres(data_df):
    K = data_df.shape[0]
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    with pm.Model() as model:
        # Extract data from dataframe
        C = data_df.shape[1] - 2
        partitions = data_df.values[:, 0:C]
        s_k = pm.ConstantData('s_k', partitions)

        # hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=10)
        σ_α = pm.HalfNormal('sigma_alpha', sigma=10)
        σ_β = pm.HalfNormal('sigma_beta', sigma=10, shape=C)

        # priors
        α = pm.Normal('alpha', mu=μ_α, sigma=σ_α, shape=K)
        β = pm.Normal('beta', mu=0, sigma=σ_β, shape=C)

        # linear model
        p = pm.math.dot(s_k, β) + α
        θ = pm.Deterministic('theta', pm.math.invlogit(p))

        # likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks)
        return model


def get_model_logres_nc(data_df):
    K = data_df.shape[0]
    ns = data_df['n_trial'].values
    ks = data_df['n_success'].values

    with pm.Model() as model:
        # Extract data from dataframe
        C = data_df.shape[1] - 2
        partitions = data_df.values[:, 0:C]
        s_k = pm.ConstantData('s_k', partitions)

        # define the standard normal random variables
        z_α = pm.Normal('z_alpha', mu=0, sigma=1, shape=K)
        z_β = pm.Normal('z_beta', mu=0, sigma=1, shape=(C,))

        # Define hyper-priors
        μ_α = pm.Normal('mu_alpha', mu=0, sigma=10)
        σ_α = pm.HalfNormal('sigma_alpha', sigma=10)
        σ_β = pm.HalfNormal('sigma_beta', sigma=10, shape=C)

        # Define priors
        α = pm.Deterministic('alpha', μ_α + (z_α * σ_α))
        β = pm.Deterministic('beta', z_β * σ_β)

        # Define the linear model using dot product
        p = pm.math.dot(s_k, β) + α
        θ = pm.Deterministic('theta', pm.math.invlogit(p))

        # Define the likelihood
        y = pm.Binomial('y', n=ns, p=θ, observed=ks)
        return model
