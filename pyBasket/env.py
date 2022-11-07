from collections import defaultdict

import arviz as az
import numpy as np
import pymc as pm
from IPython.display import display
from scipy import stats


class Group():
    '''
    Create a class to respresent a group, i.e. a basket
    '''

    GROUP_OPEN = 'OPEN'
    GROUP_EARLY_TERMINATED = 'EARLY_TERMINATED'
    GROUP_EFFECTIVE = 'EFFECTIVE'
    GROUP_INEFFECTIVE = 'INEFFECTIVE'

    def __init__(self, group_id, true_response_rate=None):
        self.idx = group_id
        self.true_response_rate = true_response_rate
        self.responses = []

    def enroll(self, num_patient):
        responses = self.sample(num_patient)
        self.responses.extend(responses)
        return self.responses

    def sample(self, num_patient):
        if self.true_response_rate is None:
            return None
        else:
            return stats.binom.rvs(1, self.true_response_rate, size=num_patient)

    @property
    def response_indices(self):
        return [self.idx] * len(self.responses)

    def __repr__(self):
        return 'Group %d (true θ=%.2f) %s' % (self.idx, self.true_response_rate, self.responses)


class Model():
    def __init__(self, ns, ks, K):
        self.ns = ns
        self.ks = ks
        self.num_groups = K
        self.model = self.model_definition()
        self.inference_data = None

    def model_definition(self):
        pass

    def infer(self, num_posterior_samples, num_burn_in):
        with self.model:
            self.inference_data = pm.sample(draws=num_posterior_samples, tune=num_burn_in,
                                            return_inferencedata=True)
            return self.inference_data

    def visualise(self):
        display(pm.model_to_graphviz(self.model))

    def plot_trace(self):
        assert self.inference_data is not None
        az.plot_trace(self.inference_data)

    def plot_posterior(self):
        assert self.inference_data is not None
        az.plot_posterior(self.inference_data)


class Independent(Model):
    def model_definition(self):
        with pm.Model() as model:
            θ = pm.Beta('θ', alpha=1, beta=1, shape=self.num_groups)
            y = pm.Binomial('y', n=self.ns, p=θ, observed=self.ks)
            return model


class Hierarchical(Model):
    def model_definition(self):
        with pm.Model() as model:
            α = pm.Gamma('α', alpha=4, beta=0.5)
            β = pm.Gamma('β', alpha=4, beta=0.5)

            θ = pm.Beta('θ', alpha=α, beta=β, shape=self.num_groups)
            y = pm.Binomial('y', n=self.ns, p=θ, observed=self.ks)

            return model


class BHM(Model):

    def __init__(self, ns, ks, K, p0):
        self.p0 = p0
        super().__init__(ns, ks, K)

    def model_definition(self):
        mu0 = np.log(self.p0 / (1 - self.p0))
        with pm.Model() as model:
            tausq = pm.Gamma('tausq', alpha=0.001, beta=0.001)
            theta_0 = pm.Normal('theta0', mu=-mu0, sigma=0.001)

            e = pm.Normal('e', mu=0, sigma=tausq, shape=self.num_groups)
            theta = pm.Deterministic('theta', theta_0 + e)
            p = pm.Deterministic('p', np.exp(theta) / (1 + np.exp(theta)))
            y = pm.Binomial('y', n=self.ns, p=p, observed=self.ks)

            return model


class Trial():
    '''
    A class to represent a clinical trial

    Can be called sequentially to enroll patients into groups. Uses pymc to infer groups' response rate

    Three models are defined inside the `get_model()` method of this class:
    - `independent`: independent Beta-Binomial models with no parameter sharing.
    - `hierarchical`: Beta-Binomial models with parameter sharing (using Gamma priors).
    - `bhm`: Normal with hierarchical priors following [Berry et al. 2013](https://journals.sagepub.com/doi/full/10.1177/1740774513497539). Ported to pymc from the JAGS implementation in https://github.com/Jin93/CBHM/blob/master/BHM.txt.
    '''

    def __init__(self, K, p0, p1, true_response_rates, enrollment,
                 evaluate_interim, num_burn_in, num_posterior_samples, model_names):
        self.K = K
        self.p0 = p0
        self.p1 = p1
        self.p_mid = (self.p0 + self.p1) / 2
        self.num_burn_in = int(num_burn_in)
        self.num_posterior_samples = int(num_posterior_samples)
        self.model_names = model_names

        assert len(true_response_rates) == K
        assert len(enrollment) == len(evaluate_interim)
        self.true_response_rates = true_response_rates
        self.enrollment = enrollment
        self.evaluate_interim = evaluate_interim

        self.current_stage = 0
        self.total_enrolled = 0
        self.inference_results = None
        self.groups = []

    def reset(self):
        self.current_stage = 0
        self.total_enrolled = 0
        self.inference_results = defaultdict(list)
        self.groups = [Group(k, self.true_response_rates[k]) for k in range(self.K)]
        return False

    def step(self):
        num_patient = self.enrollment[self.current_stage]
        self.total_enrolled += num_patient
        print('\n########## Stage=%d Enrolled = %d ##########' % (
            self.current_stage, self.total_enrolled))

        # simulate enrollment
        observed_data = []
        group_idx = []

        for group in self.groups:
            group.enroll(num_patient)
            print(group)

            observed_data.extend(group.responses)
            group_idx.extend(group.response_indices)
        print()

        if self.evaluate_interim[self.current_stage]:
            ns, ks, num_groups = self.prepare_data(group_idx, observed_data)

            # do inference at this stage
            for model_name in self.model_names:
                model = self.get_model(model_name, ns, ks, num_groups)
                model.infer(self.num_posterior_samples, self.num_burn_in)
                self.inference_results[model_name].append(model)
        else:
            for model_name in self.model_names:
                self.inference_results[model_name].append(None)

        # final termination check
        self.current_stage += 1
        return True if self.current_stage == len(self.enrollment) else False

    def get_model(self, model_name, ns, ks, num_groups):
        assert model_name in ['independent', 'hierarchical', 'bhm']
        print('\nmodel_name', model_name)
        if model_name == 'independent':
            return Independent(ns, ks, num_groups)
        elif model_name == 'hierarchical':
            return Hierarchical(ns, ks, num_groups)
        elif model_name == 'bhm':
            return BHM(ns, ks, num_groups, self.p0)

    def prepare_data(self, group_idx, observed_data):
        observed_data = np.array(observed_data)
        group_idx = np.array(group_idx)
        unique_group_idx = np.unique(group_idx)
        num_groups = len(unique_group_idx)
        ns = [len(observed_data[group_idx == idx]) for idx in unique_group_idx]
        ks = [np.sum(observed_data[group_idx == idx]) for idx in unique_group_idx]
        return ns, ks, num_groups

    def visualise_model(self, model_name):
        last_model = self.inference_results[model_name][-1]
        last_model.visualise()

    def plot_trace(self, model_name, pos):
        model = self.inference_results[model_name][pos]
        model.plot_trace()

    def plot_posterior(self, model_name, pos):
        model = self.inference_results[model_name][pos]
        model.plot_posterior()
