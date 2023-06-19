import argparse
import os
import sys

sys.path.append('..')
sys.path.append('.')

import loguru
import numpy as np
import pylab as plt
import seaborn as sns

from pyBasket.common import DEFAULT_EFFICACY_CUTOFF, DEFAULT_FUTILITY_CUTOFF, \
    MODEL_INDEPENDENT, MODEL_BHM, MODEL_PYBASKET, save_obj
from pyBasket.env import Trial, TrueResponseWithClusteringSite, TrueResponseSite


def simulate_trials(num_sim, K, p0, p1, site, evaluate_interim, num_burn_in,
                    num_posterior_samples, analysis_names, futility_cutoff,
                    efficacy_cutoff, early_futility_stop, early_efficacy_stop,
                    num_chains, pbar):
    # simulate basket trial process under the null hypothesis
    trials = []
    for i in range(num_sim):
        loguru.logger.warning(f"Running trial {i + 1}/{num_sim}")
        trial = Trial(K, p0, p1, site, evaluate_interim,
                      num_burn_in, num_posterior_samples, analysis_names,
                      futility_cutoff=futility_cutoff, efficacy_cutoff=efficacy_cutoff,
                      early_futility_stop=early_futility_stop,
                      early_efficacy_stop=early_efficacy_stop,
                      num_chains=num_chains, pbar=pbar)

        done = trial.reset()
        while not done:
            done = trial.step()
        trials.append(trial)
    return trials


def calculate_quantiles(trials, analysis_names, alpha):
    # calculate quantiles
    Qs = {}
    for analysis_name in analysis_names:
        loguru.logger.debug(f"Calculating quantiles for {analysis_name}")

        posterior_ind = []
        for trial in trials:
            probs = trial.analyses[analysis_name].df['prob'].values
            posterior_ind.append(probs)
        posterior_ind = np.array(posterior_ind)

        Q = np.quantile(posterior_ind, 1 - alpha)
        Qs[analysis_name] = (posterior_ind.flatten(), Q)
    return Qs


def get_output_filenames(args):
    if args.with_clustering_info:
        out_file = f'calibration_clustering_{args.with_clustering_info}_ncluster_{args.n_clusters}.p'
        out_plot = f'calibration_clustering_{args.with_clustering_info}_ncluster_{args.n_clusters}.png'
    else:
        out_file = f'calibration_clustering_{args.with_clustering_info}.p'
        out_plot = f'calibration_clustering_{args.with_clustering_info}.png'
    out_file = os.path.join('results', out_file)
    out_plot = os.path.join('results', out_plot)
    return out_file, out_plot


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
    parser.set_defaults(with_clustering_info=True)
    args = parser.parse_args()

    # enrollment and calibration parameters
    K = 6  # the number of groups
    p0 = 0.2  # null response rate
    p1 = 0.4  # target response rate
    enrollments = [[14, 10] for _ in range(K)]
    evaluate_interim = [True, True]  # evaluate every interim stage
    analysis_names = [MODEL_INDEPENDENT, MODEL_BHM, MODEL_PYBASKET]
    futility_cutoff = DEFAULT_FUTILITY_CUTOFF
    efficacy_cutoff = DEFAULT_EFFICACY_CUTOFF
    early_futility_stop = True
    early_efficacy_stop = False
    pbar = False

    loguru.logger.remove()  # Remove the default logger
    loguru.logger.add(sys.stderr, level='WARNING')  # Add a new logger with the desired log level

    # simulate basket trial process under the null hypothesis
    true_response_rates = [p0, p0, p0, p0, p0, p0]

    if args.with_clustering_info:
        site = TrueResponseWithClusteringSite(enrollments, K, args.n_clusters,
                                              true_response_rates=true_response_rates)
    else:
        site = TrueResponseSite(true_response_rates, enrollments)

    trials = simulate_trials(args.num_sim, K, p0, p1, site, evaluate_interim,
                             args.num_burn_in, args.num_posterior_samples,
                             analysis_names, futility_cutoff, efficacy_cutoff,
                             early_futility_stop, early_efficacy_stop, args.num_chains, pbar)

    Qs = calculate_quantiles(trials, analysis_names, args.alpha)

    out_file, out_plot = get_output_filenames(args)
    save_obj(Qs, out_file)
    plot_posteriors(Qs, analysis_names, out_plot)


if __name__ == '__main__':
    main()
