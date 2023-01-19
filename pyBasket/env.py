from abc import ABC, abstractmethod
from collections import defaultdict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from IPython.display import display
from scipy import stats

import pylab as plt
import seaborn as sns
from loguru import logger

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fclusterdata, fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as shc

from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF, DEFAULT_NUM_CHAINS, \
    GROUP_STATUS_OPEN, DEFAULT_EARLY_FUTILITY_STOP, DEFAULT_EARLY_EFFICACY_STOP, \
    GROUP_STATUS_EARLY_STOP_FUTILE, GROUP_STATUS_EARLY_STOP_EFFECTIVE, \
    GROUP_STATUS_COMPLETED_EFFECTIVE, GROUP_STATUS_COMPLETED_INEFFECTIVE, \
    MODEL_INDEPENDENT, MODEL_HIERARCHICAL, MODEL_BHM, MODEL_CLUSTERING, save_obj


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
            self.features = np.concatenate([self.features, patient_data.features], axis=0)

    @property
    def response_indices(self):
        return [self.idx] * len(self.responses)

    def __repr__(self):
        nnz = np.count_nonzero(self.responses)
        total = len(self.responses)
        return 'Group %d (%s): %d/%d' % (self.idx, self.status, nnz, total)


class ClusteringData():
    def __init__(self, groups):
        self.groups = groups

        all_features = []
        all_classes = []
        all_responses = []
        for group in self.groups:
            features = group.features
            all_features.append(features)

            N = group.features.shape[0]
            group_class = [group.idx] * N
            all_classes.extend(group_class)

            all_responses.extend(group.responses)

        self.features = np.concatenate(all_features)
        self.classes = np.array(all_classes)
        self.responses = np.array(all_responses)
        self.dist = None
        self.clusters = None

    def PCA(self, n_components=5, plot_PCA=False):
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(self.features)
        pc1_values = pcs[:, 0]
        pc2_values = pcs[:, 1]

        if plot_PCA:
            sns.set_context('poster')
            plt.figure(figsize=(5, 5))
            g = sns.scatterplot(x=pc1_values, y=pc2_values, hue=self.classes, palette='bright')
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
            plt.show()
            print('PCA explained variance', pca.explained_variance_ratio_.cumsum())

    def compute_distance_matrix(self):
        # a custom function that computes:
        # the Euclidean distance if p1 and p2 are in different baskets
        # or, returns 0 distance if p1 and p2 are in the same basket
        def mydist(p1, p2, c1, c2):
            if c1 == c2:
                return 0
            diff = p1 - p2
            return np.vdot(diff, diff) ** 0.5

        N = self.features.shape[0]
        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                p1 = self.features[i]
                p2 = self.features[j]
                c1 = self.classes[i]
                c2 = self.classes[j]
                dist[i, j] = mydist(p1, p2, c1, c2)
        self.dist = dist
        return self.dist

    def plot_distance_matrix(self):
        sns.set_context('poster')
        plt.matshow(self.dist)
        plt.colorbar()
        plt.show()

    def cluster(self, plot_dendrogram=False, max_d=70):
        condensed_dist = squareform(self.dist)
        Z = linkage(condensed_dist, method='ward')
        clusters = fcluster(Z, max_d, criterion='distance')

        if plot_dendrogram:
            plt.figure(figsize=(30, 10))
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('sample index')
            plt.ylabel('distance')
            dendrogram(
                Z,
                leaf_rotation=90.,  # rotates the x axis labels
                leaf_font_size=18,  # font size for the x axis labels
            )
            plt.ylim([-5, 120])
            plt.axhline(y=max_d, color='r', linestyle='--')
            for text in plt.gca().get_xticklabels():
                pos = int(text.get_text())
                if self.responses[pos] == 1:
                    text.set_color('blue')
                else:
                    text.set_color('red')
            plt.show()

        self.clusters = clusters
        return self.clusters

    def to_df(self):
        N = self.responses.shape[0]
        data = []
        for n in range(N):
            row = [self.responses[n], self.classes[n], self.clusters[n], self.features[n]]
            data.append(row)
        df = pd.DataFrame(data, columns=['response', 'class', 'cluster', 'features'])
        return df

    def __repr__(self):
        return 'ClusteringData: %s %s' % (str(self.features.shape), str(self.classes.shape))


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
        self.clustering_data = None

    @abstractmethod
    def model_definition(self, ns, ks):
        pass

    @abstractmethod
    def get_posterior_response(self):
        pass

    @abstractmethod
    def clustering(self, **kwargs):
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

    def clustering(self, **kwargs):
        pass

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.θ.values


class HierarchicalWithClustering(Independent):
    def clustering(self, plot_PCA=True, n_components=5, plot_distance=True, plot_dendrogram=True,
                   max_d=60):
        self.clustering_data = ClusteringData(self.groups)
        self.clustering_data.PCA(n_components=n_components, plot_PCA=plot_PCA)

        self.clustering_data.compute_distance_matrix()
        if plot_distance:
            self.clustering_data.plot_distance_matrix()

        self.clustering_data.cluster(plot_dendrogram=plot_dendrogram, max_d=max_d)


class Hierarchical(Analysis):
    def model_definition(self, ns, ks):
        with pm.Model() as model:
            α = pm.Gamma('α', alpha=4, beta=0.5)
            β = pm.Gamma('β', alpha=4, beta=0.5)

            θ = pm.Beta('θ', alpha=α, beta=β, shape=self.K)
            y = pm.Binomial('y', n=ns, p=θ, observed=ks)

            return model

    def clustering(self, **kwargs):
        pass

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

    def clustering(self, **kwargs):
        pass

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
                 num_chains=DEFAULT_NUM_CHAINS, plot_PCA=True, n_components=5,
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

            print()

        # perform clustering if necessary
        for analysis_name in self.analysis_names:
            print('Clustering for', analysis_name)
            analysis = self.analyses[analysis_name]
            analysis.clustering(plot_PCA=self.plot_PCA, n_components=self.n_components,
                                plot_distance=self.plot_distance, plot_dendrogram=self.plot_dendrogram,
                                max_d=self.max_d)

        # evaluate interim stages if needed
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

        for analysis_name in self.analysis_names:
            if self.save_analysis:
                save_obj(analysis, '%s_%d.p' % (analysis_name, self.current_stage))

        self.current_stage += 1
        return last_step

    def get_analysis(self, analysis_name, K, p0, p_mid,
                     futility_cutoff, efficacy_cutoff,
                     early_futility_stop, early_efficacy_stop):
        assert analysis_name in [MODEL_INDEPENDENT, MODEL_CLUSTERING,
                                 MODEL_HIERARCHICAL, MODEL_BHM]
        total_steps = len(self.evaluate_interim) - 1
        if analysis_name == MODEL_INDEPENDENT:
            return Independent(K, total_steps, p0, p_mid,
                               futility_cutoff, efficacy_cutoff,
                               early_futility_stop, early_efficacy_stop,
                               self.num_chains)
        if analysis_name == MODEL_CLUSTERING:
            return HierarchicalWithClustering(K, total_steps, p0, p_mid,
                                             futility_cutoff, efficacy_cutoff,
                                             early_futility_stop, early_efficacy_stop,
                                             self.num_chains)
        elif analysis_name == MODEL_HIERARCHICAL:
            return Hierarchical(K, total_steps, p0, p_mid,
                                futility_cutoff, efficacy_cutoff,
                                early_futility_stop, early_efficacy_stop,
                                self.num_chains)
        elif analysis_name == MODEL_BHM:
            return BHM(K, total_steps, p0, p_mid,
                       futility_cutoff, efficacy_cutoff,
                       early_futility_stop, early_efficacy_stop,
                       self.num_chains)

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

    def final_report(self, analysis_name):
        analysis = self.analyses[analysis_name]
        display(analysis.group_report())
