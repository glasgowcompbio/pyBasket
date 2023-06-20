import argparse
import os
import sys

from joblib import Parallel, delayed
from loguru import logger

sys.path.append('..')
sys.path.append('.')

import numpy as np
import pylab as plt
import seaborn as sns

from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF, \
    MODEL_INDEPENDENT, MODEL_INDEPENDENT_BERN, MODEL_BHM, MODEL_PYBASKET, save_obj
from pyBasket.env import Trial, TrueResponseWithClusteringSite, TrueResponseSite


class TrialResult:
    def __init__(self, analysis_results):
        self.analysis_results = analysis_results


def simulate_trial(i, num_sim, K, p0, p1, site, evaluate_interim, num_burn_in,
                   num_posterior_samples, analysis_names, futility_cutoff,
                   efficacy_cutoff, early_futility_stop, early_efficacy_stop,
                   num_chains, pbar):
    logger.warning(f"Running trial {i + 1}/{num_sim}")
    trial = Trial(K, p0, p1, site, evaluate_interim,
                  num_burn_in, num_posterior_samples, analysis_names,
                  futility_cutoff=futility_cutoff, efficacy_cutoff=efficacy_cutoff,
                  early_futility_stop=early_futility_stop,
                  early_efficacy_stop=early_efficacy_stop,
                  num_chains=num_chains, pbar=pbar)

    done = trial.reset()
    while not done:
        done = trial.step()

    # return 'prob' values for each analysis as a TrialResult object
    return TrialResult(
        {analysis_name: trial.analyses[analysis_name].df['prob'].values for analysis_name in
         analysis_names})


def simulate_trials(num_sim, K, p0, p1, site, evaluate_interim, num_burn_in,
                    num_posterior_samples, analysis_names, futility_cutoff,
                    efficacy_cutoff, early_futility_stop, early_efficacy_stop,
                    num_chains, pbar, n_jobs):
    trials = Parallel(n_jobs=n_jobs)(delayed(simulate_trial)(
        i, num_sim, K, p0, p1, site, evaluate_interim, num_burn_in,
        num_posterior_samples, analysis_names, futility_cutoff,
        efficacy_cutoff, early_futility_stop, early_efficacy_stop,
        num_chains, pbar) for i in range(num_sim))
    return trials


def calculate_quantiles(trial_results, analysis_names, alpha):
    # calculate quantiles
    Qs = {}
    for analysis_name in analysis_names:
        logger.debug(f"Calculating quantiles for {analysis_name}")
        posterior_ind = np.array(
            [result.analysis_results[analysis_name] for result in trial_results])
        Q = np.quantile(posterior_ind, 1 - alpha)
        Qs[analysis_name] = (posterior_ind.flatten(), Q)
    return Qs


def get_output_filenames(args):
    base_filename = f'calibration_clustering_{args.with_clustering_info}'
    if args.with_clustering_info:
        base_filename += f'_ncluster_{args.n_clusters}'

    out_pickle = os.path.join('results', base_filename + '.p')
    out_plot = os.path.join('results', base_filename + '.png')
    out_txt = os.path.join('results', base_filename + '.txt')

    return out_pickle, out_plot, out_txt


def plot_posteriors(Qs, analysis_names, out_plot):
    # plot posterior distributions of basket, and the quantile
    fig, axes = plt.subplots(len(analysis_names), 1, figsize=(10, len(analysis_names) * 5))
    for i, analysis_name in enumerate(analysis_names):
        ax = axes[i]
        sns.histplot(Qs[analysis_name][0], bins=30, ax=ax)
        Q = Qs[analysis_name][1]
        # draw a vertical line where Q is
        ax.axvline(x=Q, color='red', linestyle='--')  # Draw a vertical line at the position of Q
        ax.set_title(analysis_name)
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)


def save_Q(Qs, out_txt):
    with open(out_txt, 'w') as f:
        for analysis_name, (posterior_ind, Q) in Qs.items():
            f.write(f'{analysis_name}: {Q}\n')


def main():
    parser = argparse.ArgumentParser(description='PyBasket Simulation')
    parser.add_argument('--num_burn_in', type=int, default=5000,
                        help='Number of burn in iterations')
    parser.add_argument('--num_posterior_samples', type=int, default=5000,
                        help='Number of posterior samples')
    parser.add_argument('--num_chains', type=int, default=1, help='Number of chains')
    parser.add_argument('--num_sim', type=int, default=5000, help='Number of simulations')
    parser.add_argument('--n_clusters', type=int, default=1, help='Number of clusters')
    parser.add_argument('--alpha', type=float, default=0.1, help='Significance level for the test')
    parser.add_argument('--with_clustering_info', action='store_true',
                        help='Whether to include clustering information.')
    parser.add_argument('--no-with_clustering_info', dest='with_clustering_info',
                        action='store_false',
                        help='Whether to exclude clustering information.')
    parser.add_argument('--parallel', action='store_true', help='Run simulations in parallel')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of jobs for parallel execution')
    parser.set_defaults(with_clustering_info=True)
    args = parser.parse_args()

    # enrollment and calibration parameters
    K = 6  # the number of groups
    p0 = 0.2  # null response rate
    p1 = 0.4  # target response rate
    enrollments = [[14, 10] for _ in range(K)]
    evaluate_interim = [True, True]  # evaluate every interim stage
    analysis_names = [MODEL_INDEPENDENT, MODEL_INDEPENDENT_BERN, MODEL_BHM, MODEL_PYBASKET]
    futility_cutoff = DEFAULT_FUTILITY_CUTOFF
    efficacy_cutoff = DEFAULT_EFFICACY_CUTOFF
    early_futility_stop = True
    early_efficacy_stop = False
    pbar = False

    logger.remove()  # Remove the default logger
    logger.add(sys.stderr, level='WARNING')  # Add a new logger with the desired log level

    # simulate basket trial process under the null hypothesis
    true_response_rates = [p0, p0, p0, p0, p0, p0]

    if args.with_clustering_info:
        site = TrueResponseWithClusteringSite(enrollments, K, args.n_clusters,
                                              true_response_rates=true_response_rates)
    else:
        site = TrueResponseSite(true_response_rates, enrollments)

    n_jobs = args.n_jobs if args.parallel else 1
    trial_results = simulate_trials(args.num_sim, K, p0, p1, site, evaluate_interim,
                                    args.num_burn_in, args.num_posterior_samples,
                                    analysis_names, futility_cutoff, efficacy_cutoff,
                                    early_futility_stop, early_efficacy_stop, args.num_chains,
                                    pbar,
                                    n_jobs)

    # calculate quantiles
    Qs = calculate_quantiles(trial_results, analysis_names, args.alpha)

    # save the results
    out_file, out_plot, out_txt = get_output_filenames(args)
    save_obj(Qs, out_file)
    save_Q(Qs, out_txt)
    plot_posteriors(Qs, analysis_names, out_plot)


if __name__ == '__main__':
    main()
