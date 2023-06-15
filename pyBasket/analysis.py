from abc import ABC, abstractmethod

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pyBasket.common import GROUP_STATUS_EARLY_STOP_FUTILE, GROUP_STATUS_EARLY_STOP_EFFECTIVE, \
    GROUP_STATUS_COMPLETED_EFFECTIVE, GROUP_STATUS_COMPLETED_INEFFECTIVE, GROUP_STATUS_OPEN
from pyBasket.common import Group
from pyBasket.model import get_model_simple, get_model_bhm_nc, get_model_pyBasket_nc


class Analysis(ABC):
    def __init__(self, K, total_steps, p0, p_mid,
                 futility_cutoff, efficacy_cutoff,
                 early_futility_stop, early_efficacy_stop,
                 num_chains, target_accept):
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

        self.groups = [Group(k) for k in range(self.K)]
        self.df = None
        self.model = None
        self.clustering_method = None

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
        data_df = self._prepare_data(group_idx, observed_data)

        # if there are clustering results available
        if self.clustering_method is not None:
            classes = np.array(self.clustering_method.classes)
            clusters = np.array(self.clustering_method.clusters)
            # print(classes)
            # print(clusters)

            cluster_df = self.get_cluster_df(classes, clusters)
            data_df = pd.concat([cluster_df, data_df], axis=1)
        # print(data_df)

        # create model and draw posterior samples
        self.model = self.model_definition(data_df)
        with self.model:
            self.idata = pm.sample(draws=num_posterior_samples, tune=num_burn_in,
                                   chains=self.num_chains, idata_kwargs={'log_likelihood': True},
                                   target_accept=self.target_accept)

        # generate df to report the futility and efficacy
        self.df = self._check_futility_efficacy(current_step)
        return self.df

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
        ns = [len(observed_data[group_idx == idx]) for idx in unique_group_idx]
        ks = [np.sum(observed_data[group_idx == idx]) for idx in unique_group_idx]
        data_df = pd.DataFrame({
            'n_success': ks,
            'n_trial': ns
        })
        return data_df


class IndependentAnalysis(Analysis):
    def model_definition(self, data_df):
        return get_model_simple(data_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class BHMAnalysis(Analysis):
    def model_definition(self, data_df):
        return get_model_bhm_nc(data_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class PyBasketAnalysis(Analysis):
    def model_definition(self, data_df):
        return get_model_pyBasket_nc(data_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values
