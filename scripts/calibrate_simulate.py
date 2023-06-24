import argparse
import os
import sys

from joblib import Parallel, delayed
from loguru import logger

POSTERIOR_IND = 'posterior_ind'
CALIBRATED_THRESHOLD = 'calibrated_threshold'

sys.path.append('..')
sys.path.append('.')

import numpy as np
import pylab as plt
import seaborn as sns

from pyBasket.common import DEFAULT_DECISION_THRESHOLD_INTERIM, \
    MODEL_INDEPENDENT, MODEL_INDEPENDENT_BERN, MODEL_BHM, MODEL_PYBASKET, save_obj, load_obj, \
    create_if_not_exist
from pyBasket.env import Trial, TrueResponseWithClusteringSite, TrueResponseSite, TrialResult


def simulate_trial(i, num_sim, K, p0, p1, site, evaluate_interim, num_burn_in,
                   num_posterior_samples, analysis_names, dt, dt_interim,
                   early_futility_stop, num_chains, pbar):
    logger.warning(f"Running trial {i + 1}/{num_sim}")
    trial = Trial(K, p0, p1, site, evaluate_interim,
                  num_burn_in, num_posterior_samples, analysis_names,
                  dt=dt, dt_interim=dt_interim,
                  early_futility_stop=early_futility_stop, num_chains=num_chains, pbar=pbar)

    done = trial.reset()
    while not done:
        done = trial.step()

    # return 'prob' values for each analysis as a TrialResult object
    prob_values = {}
    final_reports = {}
    for analysis_name in analysis_names:
        prob_values[analysis_name] = trial.analyses[analysis_name].df['prob'].values
        final_reports[analysis_name] = trial.final_report(analysis_name)
    idfs = trial.idfs
    return TrialResult(prob_values, idfs, final_reports)


def simulate_trials(num_sim, K, p0, p1, site, evaluate_interim, num_burn_in,
                    num_posterior_samples, analysis_names, dt,
                    dt_interim, early_futility_stop,
                    num_chains, pbar, n_jobs):
    trials = Parallel(n_jobs=n_jobs)(delayed(simulate_trial)(
        i, num_sim, K, p0, p1, site, evaluate_interim, num_burn_in,
        num_posterior_samples, analysis_names, dt,
        dt_interim, early_futility_stop, num_chains, pbar) for i in range(num_sim))
    return trials


def calibrate_quantiles(trial_results, analysis_names, alpha):
    # calculate quantiles
    Qs = {}
    for analysis_name in analysis_names:
        logger.warning(f"Calculating quantiles for {analysis_name}")
        posterior_ind = np.array(
            [result.prob_values[analysis_name] for result in trial_results])
        Q = np.quantile(posterior_ind, 1 - alpha)
        Qs[analysis_name] = {
            POSTERIOR_IND: posterior_ind.flatten(),
            CALIBRATED_THRESHOLD: Q
        }
        logger.warning(Qs)
    return Qs


def get_output_filenames(result_dir, args, calibrate):
    if calibrate:
        base_filename = f'calibration_clustering_{args.with_clustering_info}'
    else:
        base_filename = f'scenario_{args.scenario}_clustering_{args.with_clustering_info}'

    if args.with_clustering_info:
        base_filename += f'_ncluster_{args.n_clusters}'

    out_plot = os.path.join(result_dir, base_filename + '.png')
    out_txt = os.path.join(result_dir, base_filename + '.txt')
    out_quantile = os.path.join(result_dir, base_filename + '_quantiles.p')
    out_trial_results = os.path.join(result_dir, base_filename + '_trial_results.p')

    return out_plot, out_txt, out_quantile, out_trial_results


def plot_posteriors(Qs, analysis_names, out_plot):
    # plot posterior distributions of basket, and the quantile
    fig, axes = plt.subplots(len(analysis_names), 1, figsize=(10, len(analysis_names) * 5))
    for i, analysis_name in enumerate(analysis_names):
        posterior_ind = Qs[analysis_name][POSTERIOR_IND]
        calibrated_threshold = Qs[analysis_name][CALIBRATED_THRESHOLD]

        ax = axes[i]
        sns.histplot(posterior_ind, bins=30, ax=ax)
        ax.axvline(x=calibrated_threshold, color='red', linestyle='--')
        ax.set_title(analysis_name)
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)


def save_Q(Qs, out_txt):
    with open(out_txt, 'w') as f:
        for analysis_name in Qs:
            Q = Qs[analysis_name][CALIBRATED_THRESHOLD]
            f.write(f'{analysis_name}: {Q}\n')


def main():
    parser = argparse.ArgumentParser(description='PyBasket Simulation')

    # MCMC parameters
    parser.add_argument('--num_burn_in', type=int, default=5000,
                        help='Number of burn in iterations')
    parser.add_argument('--num_posterior_samples', type=int, default=5000,
                        help='Number of posterior samples')
    parser.add_argument('--num_chains', type=int, default=1, help='Number of chains')
    parser.add_argument('--num_sim', type=int, default=5000, help='Number of simulations')

    # Clustering parameters
    parser.add_argument('--n_clusters', type=int, default=1, help='Number of clusters')
    parser.add_argument('--with_clustering_info', action='store_true',
                        help='Whether to include clustering information.')

    # Trial parameters
    parser.add_argument('--alpha', type=float, default=0.1, help='Significance level for the test')
    parser.add_argument('--scenario', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        help='Simulation scenario. Defaults to 0 (null).')
    parser.add_argument('--calibrate', action='store_true',
                        help='Perform calibration rather than simulation')

    # Parallel parameters
    parser.add_argument('--parallel', action='store_true', help='Run simulations in parallel')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of jobs for parallel execution')

    args = parser.parse_args()

    # Enrollment and simulation parameters
    K = 6  # the number of groups
    p0 = 0.2  # null response rate
    p1 = 0.4  # target response rate
    enrollments = [[14, 10] for _ in range(K)]
    evaluate_interim = [True, True]  # evaluate every interim stage
    analysis_names = [MODEL_INDEPENDENT, MODEL_INDEPENDENT_BERN, MODEL_BHM, MODEL_PYBASKET]
    dt_interim = DEFAULT_DECISION_THRESHOLD_INTERIM
    early_futility_stop = True
    pbar = False

    logger.remove()  # Remove the default logger
    logger.add(sys.stderr, level='WARNING')  # Add a new logger with the desired log level

    # Simulate basket trial process under the selected scenario
    true_response_rates = [p0] * K
    if args.scenario > 0:
        for i in range(args.scenario):
            true_response_rates[i] = p1

    # Generate patients for enrollment. The process will vary depending on whether we use
    # clustering or not
    site = TrueResponseSite(true_response_rates, enrollments) if not args.with_clustering_info \
        else TrueResponseWithClusteringSite(enrollments, K, args.n_clusters,
                                            true_response_rates=true_response_rates)

    n_jobs = args.n_jobs if args.parallel else 1
    result_dir = os.path.abspath('results')
    create_if_not_exist(result_dir)

    if args.calibrate:  # Calibration run

        # simulate basket trial process under the selected scenario
        dt = None
        trial_results = simulate_trials(args.num_sim, K, p0, p1, site, evaluate_interim,
                                        args.num_burn_in, args.num_posterior_samples,
                                        analysis_names, dt, dt_interim, early_futility_stop,
                                        args.num_chains, pbar, n_jobs)

        # calculate quantiles for calibration in the simulation run
        Qs = calibrate_quantiles(trial_results, analysis_names, args.alpha)

        out_plot, out_txt, out_quantile, out_trial_results = get_output_filenames(
            result_dir, args, args.calibrate)
        plot_posteriors(Qs, analysis_names, out_plot)
        save_Q(Qs, out_txt)
        save_obj(Qs, out_quantile)
        save_obj(trial_results, out_trial_results)

    else:  # Simulation run

        # load quantiles from calibration run
        _, _, calibration_quantile, _ =  get_output_filenames(result_dir, args, True)
        logger.warning(f'Loading calibration quantiles from f{calibration_quantile}')
        Qs = load_obj(calibration_quantile)

        # set decision threshold using calibrated quantiles
        dt = {analysis: Qs[analysis]['calibrated_threshold'] for analysis in analysis_names}
        logger.warning(dt)

        # simulate trials for the calibrated threshold
        trial_results = simulate_trials(args.num_sim, K, p0, p1, site, evaluate_interim,
                                        args.num_burn_in, args.num_posterior_samples,
                                        analysis_names, dt, dt_interim, early_futility_stop,
                                        args.num_chains, pbar, n_jobs)

        # save the results
        _, _, _, out_trial_results = get_output_filenames(result_dir, args, False)
        save_obj(trial_results, out_trial_results)


if __name__ == '__main__':
    main()
