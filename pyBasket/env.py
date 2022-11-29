from abc import ABC, abstractmethod
from collections import defaultdict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from IPython.display import display
from scipy import stats

from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF, DEFAULT_NUM_CHAINS, \
    GROUP_STATUS_OPEN, DEFAULT_EARLY_FUTILITY_STOP, DEFAULT_EARLY_EFFICACY_STOP, \
    GROUP_STATUS_EARLY_STOP_FUTILE, GROUP_STATUS_EARLY_STOP_EFFECTIVE, \
    GROUP_STATUS_COMPLETED_EFFECTIVE, GROUP_STATUS_COMPLETED_INEFFECTIVE


class Site(ABC):

    @abstractmethod
    def enroll(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class PatientData():
    def __init__(self, responses, features=None):
        self.responses = responses
        self.features = features

    def __repr__(self):
        return 'PatientData:\nresponses %s\nfeatures\n%s' % (self.responses, self.features)


class TrueResponseSite(Site):
    '''
    A class to represent enrollment site.
    Will continuously generate patients with each call to enroll
    '''

    def __init__(self, site_id, true_response_rate, enrollment):
        self.idx = site_id
        self.true_response_rate = true_response_rate
        self.enrollment = enrollment
        self.pos = 0

    def enroll(self):
        num_patient = self.enrollment[self.pos]
        responses = stats.binom.rvs(1, self.true_response_rate, size=num_patient)
        self.pos += 1
        return PatientData(responses)

    def reset(self):
        self.pos = 0

    def __repr__(self):
        return 'Site %d: true θ=%.2f' % (self.idx, self.true_response_rate)


class TrueResponseSiteWithFeatures(TrueResponseSite):
    def __init__(self, site_id, true_response_rate, enrollment, n, pvals):
        super().__init__(site_id, true_response_rate, enrollment)
        self.n = n
        self.pvals = pvals

    def enroll(self):
        patient_data = super().enroll()
        features = np.random.multinomial(self.n, self.pvals, size=len(patient_data.responses)).T
        return PatientData(patient_data.responses, features=features)


class EmpiricalSite(Site):

    def __init__(self, site_id, num_response, sample_size):
        self.idx = site_id
        self.num_response = num_response
        self.sample_size = sample_size
        self.responses = np.array([1] * num_response + [0] * (sample_size - num_response))

    def enroll(self):
        return PatientData(self.responses)

    def reset(self):
        pass


class Group():
    '''
    A class to represent a patient group, or basket, or arm
    '''

    def __init__(self, group_id):
        self.idx = group_id
        self.responses = []
        self.features = None
        self.status = GROUP_STATUS_OPEN

    def register(self, patient_data):
        self.responses.extend(patient_data.responses)
        if self.features is None:
            self.features = patient_data.features
        else:
            self.features = np.concatenate([self.features, patient_data.features], axis=1)

    @property
    def response_indices(self):
        return [self.idx] * len(self.responses)

    def __repr__(self):
        nnz = np.count_nonzero(self.responses)
        total = len(self.responses)
        return 'Group %d (%s): %d/%d' % (self.idx, self.status, nnz, total)


class Analysis(ABC):
    def __init__(self, K, total_steps, p0, p_mid,
                 futility_cutoff, efficacy_cutoff,
                 early_futility_stop, early_efficacy_stop,
                 num_chains):
        self.K = K
        self.total_steps = total_steps
        self.idata = None
        self.p0 = p0
        self.p_mid = p_mid
        self.futility_cutoff = futility_cutoff
        self.efficacy_cutoff = efficacy_cutoff
        self.early_futility_stop = early_futility_stop
        self.early_efficacy_stop = early_efficacy_stop
        self.num_chains = num_chains

        self.groups = [Group(k) for k in range(self.K)]
        self.df = None
        self.model = None

    @abstractmethod
    def model_definition(self, ns, ks):
        pass

    @abstractmethod
    def get_posterior_response(self):
        pass

    def infer(self, current_step, num_posterior_samples, num_burn_in):
        # prepare data in the right format for inference
        observed_data = []
        group_idx = []
        for k in range(self.K):
            group = self.groups[k]
            observed_data.extend(group.responses)
            group_idx.extend(group.response_indices)
        ns, ks, num_groups = self._prepare_data(group_idx, observed_data)

        # create model and draw posterior samples
        self.model = self.model_definition(ns, ks)
        with self.model:
            self.idata = pm.sample(draws=num_posterior_samples, tune=num_burn_in,
                                   return_inferencedata=True, chains=self.num_chains)

        # generate df to report the futility and efficacy
        self.df = self._check_futility_efficacy(current_step)
        return self.df

    def group_report(self):
        data = []
        for k in range(len(self.groups)):
            group = self.groups[k]
            nnz = np.count_nonzero(group.responses)
            total = len(group.responses)
            row = [k, group.status, nnz, total]
            data.append(row)
        columns = ['k', 'status', 'nnz', 'total']
        df = pd.DataFrame(data, columns=columns).set_index('k')
        return df

    def _check_futility_efficacy(self, current_step):
        post = self.get_posterior_response()
        data = []

        last_step = current_step == self.total_steps
        final_threshold = self.p0 if last_step else self.p_mid
        for k in range(self.K):
            group_response = post[k]
            prob = np.count_nonzero(group_response > final_threshold) / len(group_response)
            futile = prob < self.futility_cutoff if not last_step else None
            effective = prob > self.efficacy_cutoff
            row = [k, prob, futile, effective]
            data.append(row)

            if not last_step:  # update interim stage status
                if futile:
                    new_status = GROUP_STATUS_EARLY_STOP_FUTILE
                    if self.early_futility_stop:
                        self._update_open_group_status(self.groups[k], new_status)
                elif effective:
                    new_status = GROUP_STATUS_EARLY_STOP_EFFECTIVE
                    if self.early_efficacy_stop:
                        self._update_open_group_status(self.groups[k], new_status)

            else:  # final stage update
                if effective:
                    new_status = GROUP_STATUS_COMPLETED_EFFECTIVE
                else:
                    new_status = GROUP_STATUS_COMPLETED_INEFFECTIVE
                self._update_open_group_status(self.groups[k], new_status)

        columns = ['k', 'prob', 'futile', 'effective']
        df = pd.DataFrame(data, columns=columns).set_index('k')
        return df

    def _update_open_group_status(self, group, new_status):
        if group.status == GROUP_STATUS_OPEN:
            group.status = new_status

    def _prepare_data(self, group_idx, observed_data):
        observed_data = np.array(observed_data)
        group_idx = np.array(group_idx)
        unique_group_idx = np.unique(group_idx)
        num_groups = len(unique_group_idx)
        ns = [len(observed_data[group_idx == idx]) for idx in unique_group_idx]
        ks = [np.sum(observed_data[group_idx == idx]) for idx in unique_group_idx]
        return ns, ks, num_groups


class Independent(Analysis):
    def model_definition(self, ns, ks):
        with pm.Model() as model:
            θ = pm.Beta('θ', alpha=1, beta=1, shape=self.K)
            y = pm.Binomial('y', n=ns, p=θ, observed=ks)
            return model

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.θ.values


class Hierarchical(Analysis):
    def model_definition(self, ns, ks):
        with pm.Model() as model:
            α = pm.Gamma('α', alpha=4, beta=0.5)
            β = pm.Gamma('β', alpha=4, beta=0.5)

            θ = pm.Beta('θ', alpha=α, beta=β, shape=self.K)
            y = pm.Binomial('y', n=ns, p=θ, observed=ks)

            return model

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.θ.values


class BHM(Analysis):
    def model_definition(self, ns, ks):
        mu0 = np.log(self.p0 / (1 - self.p0))
        with pm.Model() as model:
            tausq = pm.Gamma('tausq', alpha=0.001, beta=0.001)
            theta_0 = pm.Normal('theta0', mu=-mu0, sigma=0.001)

            e = pm.Normal('e', mu=0, sigma=tausq, shape=self.K)
            θ = pm.Deterministic('θ', theta_0 + e)
            p = pm.Deterministic('p', np.exp(θ) / (1 + np.exp(θ)))
            y = pm.Binomial('y', n=ns, p=p, observed=ks)

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

    def __init__(self, K, p0, p1, sites,
                 evaluate_interim, num_burn_in, num_posterior_samples, analysis_names,
                 futility_cutoff=DEFAULT_FUTILITY_CUTOFF, efficacy_cutoff=DEFAULT_EFFICACY_CUTOFF,
                 early_futility_stop=DEFAULT_EARLY_FUTILITY_STOP,
                 early_efficacy_stop=DEFAULT_EARLY_EFFICACY_STOP,
                 num_chains=DEFAULT_NUM_CHAINS):
        self.K = K
        self.p0 = p0
        self.p1 = p1
        self.p_mid = (self.p0 + self.p1) / 2
        self.num_burn_in = int(num_burn_in)
        self.num_posterior_samples = int(num_posterior_samples)
        self.analysis_names = analysis_names
        self.futility_cutoff = futility_cutoff
        self.efficacy_cutoff = efficacy_cutoff
        self.early_futility_stop = early_futility_stop
        self.early_efficacy_stop = early_efficacy_stop
        self.num_chains = num_chains

        self.sites = sites
        self.evaluate_interim = evaluate_interim

        self.current_stage = 0
        self.iresults = None
        self.analyses = {}

    def reset(self):
        self.current_stage = 0
        self.iresults = defaultdict(list)

        # reset sites
        for k in range(self.K):
            self.sites[k].reset()

        # initialise all the models
        for analysis_name in self.analysis_names:
            analysis = self.get_analysis(analysis_name, self.K, self.p0, self.p_mid,
                                         self.futility_cutoff, self.efficacy_cutoff,
                                         self.early_futility_stop, self.early_efficacy_stop)
            self.analyses[analysis_name] = analysis

        return False

    def step(self):
        print('\n########## Stage=%d ##########\n' % (
            self.current_stage))

        # simulate enrollment
        for k in range(self.K):
            site = self.sites[k]
            patient_data = site.enroll()

            # register new patients to the right group in each model
            for analysis_name in self.analysis_names:
                model = self.analyses[analysis_name]
                group = model.groups[k]
                if group.status == GROUP_STATUS_OPEN:
                    group.register(patient_data)
                print('Analysis', analysis_name, group)
            print()

        # evaluate interim stages if needed
        last_step = self.current_stage == (len(self.evaluate_interim) - 1)
        if self.evaluate_interim[self.current_stage]:

            # do inference at this stage
            for analysis_name in self.analysis_names:
                print('Running inference for:', analysis_name)
                model = self.analyses[analysis_name]
                df = model.infer(self.current_stage, self.num_posterior_samples, self.num_burn_in)
                display(df)
                self.iresults[analysis_name].append(model.idata)

        self.current_stage += 1
        return last_step

    def get_analysis(self, analysis_name, K, p0, p_mid,
                     futility_cutoff, efficacy_cutoff,
                     early_futility_stop, early_efficacy_stop):
        assert analysis_name in ['independent', 'hierarchical', 'bhm']
        total_steps = len(self.evaluate_interim) - 1
        if analysis_name == 'independent':
            return Independent(K, total_steps, p0, p_mid,
                               futility_cutoff, efficacy_cutoff,
                               early_futility_stop, early_efficacy_stop,
                               self.num_chains)
        elif analysis_name == 'hierarchical':
            return Hierarchical(K, total_steps, p0, p_mid,
                                futility_cutoff, efficacy_cutoff,
                                early_futility_stop, early_efficacy_stop,
                                self.num_chains)
        elif analysis_name == 'bhm':
            return BHM(K, total_steps, p0, p_mid,
                       futility_cutoff, efficacy_cutoff,
                       early_futility_stop, early_efficacy_stop,
                       self.num_chains)

    def visualise_model(self, analysis_name):
        analysis = self.analyses[analysis_name]
        display(pm.model_to_graphviz(analysis.model))

    def plot_trace(self, analysis_name, pos):
        idata = self.iresults[analysis_name][pos]
        az.plot_trace(idata)

    def plot_posterior(self, analysis_name, pos):
        idata = self.iresults[analysis_name][pos]
        az.plot_posterior(idata)

    def final_report(self, analysis_name):
        analysis = self.analyses[analysis_name]
        display(analysis.group_report())
