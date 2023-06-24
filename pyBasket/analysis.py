from abc import ABC, abstractmethod

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pyBasket.common import GROUP_STATUS_EARLY_STOP_FUTILE, GROUP_STATUS_COMPLETED_EFFECTIVE, \
    GROUP_STATUS_COMPLETED_INEFFECTIVE, GROUP_STATUS_OPEN
from pyBasket.common import Group
from pyBasket.model import get_model_simple, get_model_bhm_nc, get_model_pyBasket_nc, \
    get_model_simple_bern, get_model_hierarchical_bern


class Analysis(ABC):
    def __init__(self, K, total_steps, p0, p_mid, p1,
                 dt, dt_interim, early_futility_stop,
                 num_chains, target_accept, progress_bar):
        self.K = K
        self.total_steps = total_steps
        self.idata = None
        self.p0 = p0
        self.p_mid = p_mid
        self.p1 = p1
        self.dt = dt
        self.dt_interim = dt_interim
        self.early_futility_stop = early_futility_stop
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
        self.model = self.model_definition(count_df, obs_df, self.p0, self.p1)
        with self.model:
            self.idata = pm.sample(draws=num_posterior_samples, tune=num_burn_in,
                                   chains=self.num_chains, idata_kwargs={'log_likelihood': True},
                                   target_accept=self.target_accept, progressbar=self.pbar)

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
        return self.df

    def _check_futility_efficacy(self, current_step):
        post = self.get_posterior_response()
        data = []

        last_step = current_step == self.total_steps
        threshold = self.p0 if last_step else self.p_mid
        for k in range(self.K):
            group = self.groups[k]
            group_response = post[k]
            prob = np.count_nonzero(group_response > threshold) / len(group_response)
            effective = None

            if not last_step:  # update interim stage status
                Q = self.dt_interim
                assert Q is not None

                effective = (prob >= Q)
                if not effective and self.early_futility_stop:
                    new_status = GROUP_STATUS_EARLY_STOP_FUTILE
                    self._update_open_group_status(group, new_status)

            else:  # final stage update
                Q = self.dt
                if Q is None:
                    # for calibration step, no dt is provided, so no status is needed either
                    new_status = None
                else:
                    # for simulation step, we do the usual significance check
                    effective = (prob >= Q)
                    if effective:
                        new_status = GROUP_STATUS_COMPLETED_EFFECTIVE
                    else:
                        new_status = GROUP_STATUS_COMPLETED_INEFFECTIVE
                self._update_open_group_status(group, new_status)

            row = [k, prob, Q, effective, group.status, group.nnz, group.total]
            data.append(row)

        columns = ['k', 'prob', 'Q', 'effective', 'group_status', 'group_nnz', 'group_total']
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
    def model_definition(self, count_df, obs_df, p0, p1):
        return get_model_simple(count_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class IndependentBernAnalysis(Analysis):
    def model_definition(self, count_df, obs_df, p0, p1):
        return get_model_simple_bern(obs_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class HierarchicalBernAnalysis(Analysis):
    def model_definition(self, count_df, obs_df, p0, p1):
        return get_model_hierarchical_bern(obs_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class BHMAnalysis(Analysis):
    def model_definition(self, count_df, obs_df, p0, p1):
        return get_model_bhm_nc(count_df, p0, p1)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values


class PyBasketAnalysis(Analysis):
    def model_definition(self, count_df, obs_df, p0, p1):
        return get_model_pyBasket_nc(obs_df)

    def get_posterior_response(self):
        stacked = az.extract(self.idata)
        return stacked.basket_p.values
