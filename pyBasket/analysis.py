from abc import ABC, abstractmethod

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pyBasket.common import GROUP_STATUS_EARLY_STOP_FUTILE, GROUP_STATUS_EARLY_STOP_EFFECTIVE, \
    GROUP_STATUS_COMPLETED_EFFECTIVE, GROUP_STATUS_COMPLETED_INEFFECTIVE, GROUP_STATUS_OPEN
from pyBasket.common import Group
from pyBasket.model import get_model_simple, get_model_bhm_nc, get_model_pyBasket_nc, \
    get_model_simple_bern


class Analysis(ABC):
    def __init__(self, K, total_steps, p0, p_mid,
                 futility_cutoff, efficacy_cutoff,
                 early_futility_stop, early_efficacy_stop,
                 num_chains, target_accept, progress_bar):
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
        self.target_accept = target_accept
        self.pbar = progress_bar

        self.groups = [Group(k) for k in range(self.K)]
        self.df = None
        self.model = None
        self.clustering_method = None

    @abstractmethod
    def get_posterior_response(self):
        pass

    def infer(self, current_step, num_posterior_samples, num_burn_in):
        # prepare count data
        responses = []
        classes = []
        clusters = []
        for k in range(self.K):
            group = self.groups[k]
            responses.extend(group.responses)
            classes.extend(group.classes)
            clusters.extend(group.clusters)
        count_df = self._prepare_data(classes, responses)

        # prepare individual observation data
        if len(clusters) > 0:
            obs_df = pd.DataFrame({
                'basket_number': classes,
                'cluster_number': clusters,
                'responsive': responses
            })
        else:
            obs_df = pd.DataFrame({
                'basket_number': classes,
                'responsive': responses
            })

        # create model and draw posterior samples
        self.model = self.model_definition(count_df, obs_df)
        with self.model:
            self.idata = pm.sample(draws=num_posterior_samples, tune=num_burn_in,
                                   chains=self.num_chains, idata_kwargs={'log_likelihood': True},
                                   target_accept=self.target_accept, progressbar=self.pbar)

        # generate df to report the futility and efficacy
        self.df = self._check_futility_efficacy(current_step)
        return self.df.copy()

    def get_cluster_df(self, class_labels, cluster_labels):
        unique_clusters = np.unique(cluster_labels)
        unique_classes = np.unique(class_labels)
        cluster_df = pd.DataFrame(index=unique_classes)

        for i, cluster in enumerate(unique_clusters):
            cluster_members = class_labels[cluster_labels == cluster]
            unique_class_counts, counts = np.unique(cluster_members, return_counts=True)
            proportion = counts / cluster_members.shape[0]
            cluster_df['sk_' + str(i)] = pd.Series(dict(zip(unique_class_counts, proportion)))

        cluster_df = cluster_df.fillna(0)
        return cluster_df

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
            effective = prob >= self.efficacy_cutoff
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

    def _prepare_data(self, classes, responses):
        responses = np.array(responses)
        classes = np.array(classes)
        unique_classes = np.unique(classes)
        ns = [len(responses[classes == idx]) for idx in unique_classes]
        ks = [np.sum(responses[classes == idx]) for idx in unique_classes]
        data_df = pd.DataFrame({
            'n_success': ks,
            'n_trial': ns
        })
        return data_df


class IndependentAnalysis(Analysis):
    def model_definition(self, count_df, obs_df):
        return get_model_simple(count_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class IndependentBernAnalysis(Analysis):
    def model_definition(self, count_df, obs_df):
        return get_model_simple_bern(obs_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class BHMAnalysis(Analysis):
    def model_definition(self, count_df, obs_df):
        return get_model_bhm_nc(count_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class PyBasketAnalysis(Analysis):
    def model_definition(self, count_df, obs_df):
        return get_model_pyBasket_nc(obs_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values
