from collections import defaultdict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from IPython.display import display
from scipy import stats

from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF, DEFAULT_NUM_CHAINS


class Site():
    '''
    A class to represent enrollment site.
    Will continuously generate patients with each call to enroll
    '''

    def __init__(self, site_id, true_response_rate):
        self.idx = site_id
        self.true_response_rate = true_response_rate

    def enroll(self, num_patient):
        responses = stats.binom.rvs(1, self.true_response_rate, size=num_patient)
        return responses

    def __repr__(self):
        return 'Site %d: true θ=%.2f' % (self.idx, self.true_response_rate)


class Group():
    '''
    A class to represent a patient group, or basket, or arm
    '''

    def __init__(self, group_id):
        self.idx = group_id
        self.responses = []

    def register(self, responses):
        self.responses.extend(responses)
        return self.responses

    @property
    def response_indices(self):
        return [self.idx] * len(self.responses)

    def __repr__(self):
        return 'Group %d: %s' % (self.idx, self.responses)





class Model():
    def __init__(self, idx, ns, ks, K, p0, p_mid, futility_cutoff, efficacy_cutoff, last_step,
                 num_chains):
        self.idx = idx
        self.ns = ns
        self.ks = ks
        self.num_groups = K
        self.idata = None
        self.p0 = p0
        self.p_mid = p_mid
        self.last_step = last_step
        self.futility_cutoff = futility_cutoff
        self.efficacy_cutoff = efficacy_cutoff
        self.num_chains = num_chains
        self.df = None
        self.model = self.model_definition()

    def model_definition(self):
        pass

    def infer(self, num_posterior_samples, num_burn_in):
        with self.model:
            self.idata = pm.sample(draws=num_posterior_samples, tune=num_burn_in,
                                   return_inferencedata=True, chains=self.num_chains)
        self.df = self.check_futility_efficacy()

    def visualise(self):
        display(pm.model_to_graphviz(self.model))

    def plot_trace(self):
        assert self.idata is not None
        az.plot_trace(self.idata)

    def plot_posterior(self):
        assert self.idata is not None
        az.plot_posterior(self.idata)

    def check_futility_efficacy(self):
        post = self.get_posterior_response()
        data = []
        to_check = self.p0 if self.last_step else self.p_mid
        for k in range(self.num_groups):
            group_response = post[k]
            prob = np.count_nonzero(group_response > to_check) / len(group_response)
            futile = prob < DEFAULT_FUTILITY_CUTOFF if not self.last_step else None
            effective = prob > DEFAULT_EFFICACY_CUTOFF
            row = [k, prob, futile, effective]
            data.append(row)

        columns = ['k', 'prob', 'futile', 'effective']
        df = pd.DataFrame(data, columns=columns).set_index('k')
        return df

    def get_posterior_response(self):
        pass


class Independent(Model):
    def model_definition(self):
        with pm.Model() as model:
            θ = pm.Beta('θ', alpha=1, beta=1, shape=self.num_groups)
            y = pm.Binomial('y', n=self.ns, p=θ, observed=self.ks)
            return model

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.θ.values


class Hierarchical(Model):
    def model_definition(self):
        with pm.Model() as model:
            α = pm.Gamma('α', alpha=4, beta=0.5)
            β = pm.Gamma('β', alpha=4, beta=0.5)

            θ = pm.Beta('θ', alpha=α, beta=β, shape=self.num_groups)
            y = pm.Binomial('y', n=self.ns, p=θ, observed=self.ks)

            return model

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.θ.values


class BHM(Model):
    def model_definition(self):
        mu0 = np.log(self.p0 / (1 - self.p0))
        with pm.Model() as model:
            tausq = pm.Gamma('tausq', alpha=0.001, beta=0.001)
            theta_0 = pm.Normal('theta0', mu=-mu0, sigma=0.001)

            e = pm.Normal('e', mu=0, sigma=tausq, shape=self.num_groups)
            θ = pm.Deterministic('θ', theta_0 + e)
            p = pm.Deterministic('p', np.exp(θ) / (1 + np.exp(θ)))
            y = pm.Binomial('y', n=self.ns, p=p, observed=self.ks)

            return model

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.p.values


class Trial():
    '''
    A class to represent a clinical trial

    Can be called sequentially to enroll patients into groups. Uses pymc to infer groups'
    response rate

    Three models are defined inside the `get_model()` method of this class:
    - `independent`: independent Beta-Binomial models with no parameter sharing.
    - `hierarchical`: Beta-Binomial models with parameter sharing (using Gamma priors).
    - `bhm`: Normal with hierarchical priors following
    [Berry et al. 2013](https://journals.sagepub.com/doi/full/10.1177/1740774513497539).
    Ported from the JAGS implementation in https://github.com/Jin93/CBHM/blob/master/BHM.txt.
    '''

    def __init__(self, K, p0, p1, true_response_rates, enrollment,
                 evaluate_interim, num_burn_in, num_posterior_samples, model_names,
                 futility_cutoff=DEFAULT_FUTILITY_CUTOFF, efficacy_cutoff=DEFAULT_EFFICACY_CUTOFF,
                 num_chains=DEFAULT_NUM_CHAINS):
        self.K = K
        self.p0 = p0
        self.p1 = p1
        self.p_mid = (self.p0 + self.p1) / 2
        self.num_burn_in = int(num_burn_in)
        self.num_posterior_samples = int(num_posterior_samples)
        self.model_names = model_names
        self.futility_cutoff = futility_cutoff
        self.efficacy_cutoff = efficacy_cutoff
        self.num_chains = num_chains

        assert len(true_response_rates) == K
        assert len(enrollment) == len(evaluate_interim)
        self.true_response_rates = true_response_rates
        self.enrollment = enrollment
        self.evaluate_interim = evaluate_interim

        self.current_stage = 0
        self.total_enrolled = 0
        self.iresults = None
        self.sites = []
        self.groups = []

    def reset(self):
        self.current_stage = 0
        self.total_enrolled = 0
        self.iresults = defaultdict(list)

        # initialise K sites and K groups
        for k in range(self.K):
            site = Site(k, self.true_response_rates[k])
            group = Group(k)
            self.sites.append(site)
            self.groups.append(group)

        return False

    def step(self):
        num_patient = self.enrollment[self.current_stage]
        self.total_enrolled += num_patient
        print('\n########## Stage=%d Enrolled = %d ##########' % (
            self.current_stage, self.total_enrolled))

        # simulate enrollment
        observed_data = []
        group_idx = []
        for k in range(self.K):
            site = self.sites[k]
            group = self.groups[k]
            responses = site.enroll(num_patient)
            group.register(responses)
            print(group)

            observed_data.extend(group.responses)
            group_idx.extend(group.response_indices)
        print()

        # evaluate interim stages if needed
        last_step = self.current_stage == (len(self.enrollment) - 1)
        if self.evaluate_interim[self.current_stage]:
            ns, ks, num_groups = self.prepare_data(group_idx, observed_data)

            # do inference at this stage
            for model_name in self.model_names:
                model = self.get_model(self.current_stage, model_name, ns, ks, num_groups, self.p0,
                                       self.p_mid, self.futility_cutoff, self.efficacy_cutoff,
                                       last_step)
                model.infer(self.num_posterior_samples, self.num_burn_in)
                display(model.df)
                self.iresults[model_name].append(model)

        self.current_stage += 1
        return last_step

    def prepare_data(self, group_idx, observed_data):
        observed_data = np.array(observed_data)
        group_idx = np.array(group_idx)
        unique_group_idx = np.unique(group_idx)
        num_groups = len(unique_group_idx)
        ns = [len(observed_data[group_idx == idx]) for idx in unique_group_idx]
        ks = [np.sum(observed_data[group_idx == idx]) for idx in unique_group_idx]
        return ns, ks, num_groups

    def get_model(self, idx, model_name, ns, ks, num_groups,
                  p0, p_mid, futility_cutoff, efficacy_cutoff, last_step):
        assert model_name in ['independent', 'hierarchical', 'bhm']
        print('\nmodel_name', model_name)
        if model_name == 'independent':
            return Independent(idx, ns, ks, num_groups, p0, p_mid, futility_cutoff,
                               efficacy_cutoff, last_step, self.num_chains)
        elif model_name == 'hierarchical':
            return Hierarchical(idx, ns, ks, num_groups, p0, p_mid, futility_cutoff,
                                efficacy_cutoff, last_step, self.num_chains)
        elif model_name == 'bhm':
            return BHM(idx, ns, ks, num_groups, p0, p_mid, futility_cutoff, efficacy_cutoff,
                       last_step, self.num_chains)

    def visualise_model(self, model_name):
        last_model = self.iresults[model_name][-1]
        last_model.visualise()

    def plot_trace(self, model_name, pos):
        model = self.iresults[model_name][pos]
        model.plot_trace()

    def plot_posterior(self, model_name, pos):
        model = self.iresults[model_name][pos]
        model.plot_posterior()
