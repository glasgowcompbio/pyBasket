from abc import ABC, abstractmethod
from collections import defaultdict

import arviz as az
import numpy as np
import pymc as pm
from IPython.display import display
from loguru import logger
from scipy import stats
from scipy.special import expit as logistic
from scipy.stats import halfnorm

from pyBasket.analysis import IndependentAnalysis, BHMAnalysis, PyBasketAnalysis
from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF, DEFAULT_NUM_CHAINS, \
    GROUP_STATUS_OPEN, DEFAULT_EARLY_FUTILITY_STOP, DEFAULT_EARLY_EFFICACY_STOP, \
    MODEL_INDEPENDENT, MODEL_BHM, MODEL_PYBASKET, DEFAULT_TARGET_ACCEPT


class PatientData():
    def __init__(self, responses, classes, clusters):
        self.responses = responses
        self.classes = classes
        self.clusters = clusters

    def __repr__(self):
        return 'PatientData:\nresponses %s\nclasses %s\nclusters %s' % (
            self.responses, self.classes, self.clusters)


class Site(ABC):

    @abstractmethod
    def enroll(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class TrueResponseSite(Site):
    '''
    A class to represent the enrollment site.
    Will continuously generate patients with each call to enroll
    '''

    def __init__(self, true_response_rates, enrollments):
        self.true_response_rates = true_response_rates
        self.enrollments = enrollments
        self.pos = [0] * len(true_response_rates)

    def enroll(self, k):
        try:
            num_patient = self.enrollments[k][self.pos[k]]
        except IndexError:
            logger.warning('Enrollment has been closed for this site')
            return

        responses = stats.binom.rvs(1, self.true_response_rates[k], size=num_patient)
        classes = [k] * len(responses)
        clusters = [0] * len(responses) # put all data points into the same cluster
        self.pos[k] += 1
        return PatientData(responses, classes, clusters)

    def reset(self):
        self.pos = [0] * len(self.true_response_rates)

    def __repr__(self):
        return 'Site with true θ=%s' % (self.true_response_rates,)


class TrueResponseWithClusteringSite(Site):
    '''
    A class to represent the enrollment site.
    It will continuously generate patients with each call to enroll
    '''

    def __init__(self, enrollments, n_classes, n_clusters, true_response_rates=None):
        # The enrollment list for each group
        self.enrollments = enrollments
        # The current position in the enrollment list for each group
        self.pos = [0] * n_classes
        # The number of classes
        self.n_classes = n_classes
        # The number of clusters
        self.n_clusters = n_clusters

        # Generate unique prior mean and std for each column in theta_cluster
        self.prior_means = np.random.normal(loc=0, scale=2, size=self.n_clusters)
        # Parameters for the half-normal distribution used to generate prior standard deviations
        prior_std_mean = 0
        prior_std_std = 1
        prior_std_scale = np.sqrt(2) * prior_std_std / np.pi
        self.prior_stds = halfnorm.rvs(loc=prior_std_mean, scale=prior_std_scale,
                                       size=self.n_clusters)

        # Class and cluster indices
        self.basket_coords = np.arange(n_classes)
        self.cluster_coords = np.arange(self.n_clusters)

        # Generate or assign the true response rates for each basket (class)
        if true_response_rates is None:
            # If not provided, sample from the prior
            theta_basket = np.random.normal(loc=0, scale=2, size=n_classes)
            self.true_basket_p = logistic(theta_basket)
        else:
            # If provided, use the provided values
            assert len(true_response_rates) == self.n_classes
            self.true_basket_p = np.array(true_response_rates)  # Convert to a numpy array

        # Generate the true response rates for each cluster based on the prior means and standard deviations
        self.theta_cluster = np.zeros((n_classes, self.n_clusters))
        for i in range(self.n_clusters):
            self.theta_cluster[:, i] = np.random.normal(loc=self.prior_means[i],
                                                        scale=self.prior_stds[i], size=n_classes)

        self.true_cluster_p = logistic(self.theta_cluster)
        # Compute the true response rate for each combination of basket and cluster
        self.true_mat = self.true_basket_p.reshape((n_classes, 1)) * self.true_cluster_p

    def enroll(self, group_idx):
        try:
            # Number of patients to enroll for the group
            num_patient = self.enrollments[group_idx][self.pos[group_idx]]
        except IndexError:
            # Handle the case when there are no more patients to enroll for the group
            logger.warning(f'Enrollment has been closed for group {group_idx}')
            return

        # Randomly assign each patient to a cluster
        cluster_idx = np.random.choice(self.cluster_coords, size=num_patient)
        # Compute the true response rate for each patient based on their basket and cluster
        true_patient_p = self.true_mat[group_idx, cluster_idx]
        # Sample the response of each patient from a binomial distribution
        is_responsive = np.random.binomial(n=1, p=true_patient_p)

        # The class of each patient is the group they belong to
        classes = [group_idx] * num_patient
        # The cluster of each patient
        clusters = list(cluster_idx)

        # Move to the next position in the enrollment list for the group
        self.pos[group_idx] += 1

        return PatientData(responses=is_responsive, classes=classes, clusters=clusters)

    def reset(self):
        # Reset the current position in the enrollment list for all groups
        self.pos = [0] * self.n_classes

    def __repr__(self):
        return 'TrueResponseWithClusteringSite'


class EmpiricalSite(Site):

    def __init__(self, site_names, num_responses, sample_sizes):
        self.site_names = site_names
        self.num_responses = num_responses
        self.sample_sizes = sample_sizes

    def enroll(self, k):
        responses = self._get_responses(self.num_responses[k], self.sample_sizes[k])
        classes = [k] * len(responses)
        clusters = []
        return PatientData(responses, classes, clusters)

    def _get_responses(self, num_response, sample_size):
        return np.array([1] * num_response + [0] * (sample_size - num_response))

    def reset(self):
        pass

    def __repr__(self):
        return f'EmpiricalSite for {self.site_names}'


class Trial():
    '''
    A class to represent a clinical trial

    Can be called sequentially to enroll patients into groups. Uses pymc to infer groups'
    response rate

    Three models are defined inside the `get_analysis()` method of this class:
    - `simple`: Beta-Binomial models with parameter sharing (using Gamma priors).
    - `bhm`: Normal with hierarchical priors following
    [Berry et al. 2013](https://journals.sagepub.com/doi/full/10.1177/1740774513497539).
    - 'pyBasket': BHM-like model with extra parameters for cluster memberships
    '''

    def __init__(self, K, p0, p1, site,
                 evaluate_interim, num_burn_in, num_posterior_samples, analysis_names,
                 futility_cutoff=DEFAULT_FUTILITY_CUTOFF, efficacy_cutoff=DEFAULT_EFFICACY_CUTOFF,
                 early_futility_stop=DEFAULT_EARLY_FUTILITY_STOP,
                 early_efficacy_stop=DEFAULT_EARLY_EFFICACY_STOP,
                 num_chains=DEFAULT_NUM_CHAINS, target_accept=DEFAULT_TARGET_ACCEPT,
                 save_analysis=False, pbar=False):
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

        self.site = site
        self.evaluate_interim = evaluate_interim
        self.save_analysis = save_analysis

        self.current_stage = 0
        self.iresults = None
        self.analyses = {}
        self.pbar = pbar

    def reset(self):
        self.current_stage = 0
        self.iresults = defaultdict(list)

        # reset sites
        for k in range(self.K):
            self.site.reset()

        # initialise all the models
        for analysis_name in self.analysis_names:
            analysis = self.get_analysis(analysis_name, self.K, self.p0, self.p_mid,
                                         self.futility_cutoff, self.efficacy_cutoff,
                                         self.early_futility_stop, self.early_efficacy_stop,
                                         self.pbar)
            self.analyses[analysis_name] = analysis

        return False

    def step(self):
        logger.debug('\n########## Stage=%d ##########\n' % (
            self.current_stage))

        # simulate enrollment
        for k in range(self.K):
            patient_data = self.site.enroll(k)

            # register new patients to the right group in each model
            for analysis_name in self.analysis_names:
                analysis = self.analyses[analysis_name]
                group = analysis.groups[k]
                if group.status == GROUP_STATUS_OPEN:
                    group.register(patient_data)
                logger.debug(f'Registering {group} for Analysis {analysis_name}')

        # evaluate interim stages if needed
        logger.debug('\n')
        last_step = self.current_stage == (len(self.evaluate_interim) - 1)
        if self.evaluate_interim[self.current_stage]:

            # do inference at this stage
            for analysis_name in self.analysis_names:
                logger.debug(f'Running inference for analysis_name')
                analysis = self.analyses[analysis_name]
                analysis.infer(self.current_stage, self.num_posterior_samples,
                               self.num_burn_in)
                self.iresults[analysis_name].append(analysis.idata)

        # for analysis_name in self.analysis_names:
        #     if self.save_analysis:
        #         save_obj(analysis, '%s_%d.p' % (analysis_name, self.current_stage))

        self.current_stage += 1
        return last_step

    def get_analysis(self, analysis_name, K, p0, p_mid,
                     futility_cutoff, efficacy_cutoff,
                     early_futility_stop, early_efficacy_stop, pbar):
        assert analysis_name in [MODEL_INDEPENDENT, MODEL_BHM, MODEL_PYBASKET]
        total_steps = len(self.evaluate_interim) - 1
        if analysis_name == MODEL_INDEPENDENT:
            return IndependentAnalysis(K, total_steps, p0, p_mid,
                                       futility_cutoff, efficacy_cutoff,
                                       early_futility_stop, early_efficacy_stop,
                                       self.num_chains, self.target_accept, pbar)
        elif analysis_name == MODEL_BHM:
            return BHMAnalysis(K, total_steps, p0, p_mid,
                               futility_cutoff, efficacy_cutoff,
                               early_futility_stop, early_efficacy_stop,
                               self.num_chains, self.target_accept, pbar)
        elif analysis_name == MODEL_PYBASKET:
            return PyBasketAnalysis(K, total_steps, p0, p_mid,
                                    futility_cutoff, efficacy_cutoff,
                                    early_futility_stop, early_efficacy_stop,
                                    self.num_chains, self.target_accept, pbar)

    def visualise_model(self, analysis_name):
        try:
            analysis = self.analyses[analysis_name]
            display(pm.model_to_graphviz(analysis.model))
        except TypeError:
            logger.warning('No model to visualise')

    def plot_trace(self, analysis_name, pos):
        try:
            idata = self.iresults[analysis_name][pos]
            az.plot_trace(idata)
        except IndexError:
            logger.warning('No model to visualise')

    def plot_posterior(self, analysis_name, pos):
        try:
            idata = self.iresults[analysis_name][pos]
            az.plot_posterior(idata)
        except IndexError:
            logger.warning('No model to visualise')

    def plot_forest(self, analysis_name, pos):
        try:
            idata = self.iresults[analysis_name][pos]
            az.plot_forest(idata, var_names='basket_p')
        except IndexError:
            logger.warning('No model to visualise')

    def final_report(self, analysis_name):
        analysis = self.analyses[analysis_name]
        display(analysis.group_report())
