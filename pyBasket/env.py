from abc import ABC, abstractmethod
from collections import defaultdict

import arviz as az
import numpy as np
import pymc as pm
from IPython.display import display
from scipy import stats

from pyBasket.model import Simple, BHM, LogisticRegression
from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF, DEFAULT_NUM_CHAINS, \
    GROUP_STATUS_OPEN, DEFAULT_EARLY_FUTILITY_STOP, DEFAULT_EARLY_EFFICACY_STOP, \
    MODEL_SIMPLE, MODEL_BHM, MODEL_LOGRES, save_obj, DEFAULT_TARGET_ACCEPT


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
        features = np.random.multinomial(self.n, self.pvals, size=len(patient_data.responses))
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


class Trial():
    '''
    A class to represent a clinical trial

    Can be called sequentially to enroll patients into groups. Uses pymc to infer groups'
    response rate

    Three models are defined inside the `get_analysis()` method of this class:
    - `simple`: Beta-Binomial models with parameter sharing (using Gamma priors).
    - `bhm`: Normal with hierarchical priors following
    [Berry et al. 2013](https://journals.sagepub.com/doi/full/10.1177/1740774513497539).
    - 'logres': BHM model with extra parameters for cluster memberships
    '''

    def __init__(self, K, p0, p1, sites,
                 evaluate_interim, num_burn_in, num_posterior_samples, analysis_names,
                 futility_cutoff=DEFAULT_FUTILITY_CUTOFF, efficacy_cutoff=DEFAULT_EFFICACY_CUTOFF,
                 early_futility_stop=DEFAULT_EARLY_FUTILITY_STOP,
                 early_efficacy_stop=DEFAULT_EARLY_EFFICACY_STOP,
                 num_chains=DEFAULT_NUM_CHAINS, target_accept=DEFAULT_TARGET_ACCEPT,
                 plot_PCA=True, n_components=5,
                 plot_distance=True, plot_dendrogram=True, max_d=60,
                 save_analysis=False):
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
        self.target_accept = target_accept

        # clustering params
        self.plot_PCA = plot_PCA
        self.n_components = n_components
        self.plot_distance = plot_distance
        self.plot_dendrogram = plot_dendrogram
        self.max_d = max_d

        self.sites = sites
        self.evaluate_interim = evaluate_interim
        self.save_analysis = save_analysis

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
                analysis = self.analyses[analysis_name]
                group = analysis.groups[k]
                if group.status == GROUP_STATUS_OPEN:
                    group.register(patient_data)
                print('Registering', group, 'for Analysis', analysis_name)

        # perform clustering if necessary
        print()
        for analysis_name in self.analysis_names:
            print('Clustering for', analysis_name)
            analysis = self.analyses[analysis_name]
            analysis.clustering(plot_PCA=self.plot_PCA, n_components=self.n_components,
                                plot_distance=self.plot_distance, plot_dendrogram=self.plot_dendrogram,
                                max_d=self.max_d)

        # evaluate interim stages if needed
        print()
        last_step = self.current_stage == (len(self.evaluate_interim) - 1)
        if self.evaluate_interim[self.current_stage]:

            # do inference at this stage
            for analysis_name in self.analysis_names:
                print('Running inference for:', analysis_name)
                analysis = self.analyses[analysis_name]
                df = analysis.infer(self.current_stage, self.num_posterior_samples,
                                    self.num_burn_in)
                display(df)
                self.iresults[analysis_name].append(analysis.idata)

        # for analysis_name in self.analysis_names:
        #     if self.save_analysis:
        #         save_obj(analysis, '%s_%d.p' % (analysis_name, self.current_stage))

        self.current_stage += 1
        return last_step

    def get_analysis(self, analysis_name, K, p0, p_mid,
                     futility_cutoff, efficacy_cutoff,
                     early_futility_stop, early_efficacy_stop):
        assert analysis_name in [MODEL_SIMPLE, MODEL_BHM, MODEL_LOGRES]
        total_steps = len(self.evaluate_interim) - 1
        if analysis_name == MODEL_SIMPLE:
            return Simple(K, total_steps, p0, p_mid,
                          futility_cutoff, efficacy_cutoff,
                          early_futility_stop, early_efficacy_stop,
                          self.num_chains, self.target_accept)
        elif analysis_name == MODEL_BHM:
            return BHM(K, total_steps, p0, p_mid,
                       futility_cutoff, efficacy_cutoff,
                       early_futility_stop, early_efficacy_stop,
                       self.num_chains, self.target_accept)
        elif analysis_name == MODEL_LOGRES:
            return LogisticRegression(K, total_steps, p0, p_mid,
                                      futility_cutoff, efficacy_cutoff,
                                      early_futility_stop, early_efficacy_stop,
                                      self.num_chains, self.target_accept)

    def visualise_model(self, analysis_name):
        try:
            analysis = self.analyses[analysis_name]
            display(pm.model_to_graphviz(analysis.model))
        except TypeError:
            print('No model to visualise')

    def plot_trace(self, analysis_name, pos):
        try:
            idata = self.iresults[analysis_name][pos]
            az.plot_trace(idata)
        except IndexError:
            print('No model to visualise')

    def plot_posterior(self, analysis_name, pos):
        try:
            idata = self.iresults[analysis_name][pos]
            az.plot_posterior(idata)
        except IndexError:
            print('No model to visualise')

    def plot_forest(self, analysis_name, pos):
        try:
            idata = self.iresults[analysis_name][pos]
            az.plot_forest(idata, var_names='theta')
        except IndexError:
            print('No model to visualise')

    def final_report(self, analysis_name):
        analysis = self.analyses[analysis_name]
        display(analysis.group_report())
