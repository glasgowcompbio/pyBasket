import numpy as np
import pylab as plt
import socket

from vimms.Evaluation import evaluate_simulated_env

MS1_REWARD = 0.1
INVALID_MOVE_REWARD = -1.0
NO_FRAGMENTATION_REWARD = -100.0
MAX_OBSERVED_INTENSITY = 1E20
MAX_OBSERVED_LOG_INTENSITY = np.log(MAX_OBSERVED_INTENSITY)
MAX_ROI_LENGTH_SECONDS = 100
HISTORY_HORIZON = 1
MAX_EVAL_TIME_PER_EPISODE = 300
EVAL_F1_INTENSITY_THRESHOLD = 0.5

ALPHA = 0.5
BETA = 0

METHOD_RANDOM = 'random'
METHOD_FULLSCAN = 'fullscan'
METHOD_TOPN = 'topN'
METHOD_PPO = 'PPO'
METHOD_PPO_RECURRENT = 'RecurrentPPO'
METHOD_DQN = 'DQN'
METHOD_DQN_COV = 'DQN_COV'
METHOD_DQN_INT = 'DQN_INT'
METHOD_DQN_MID = 'DQN_MID'

GYM_ENV_NAME = 'DDAEnv'
if socket.gethostname() == 'Macbook-Air.local':
    GYM_NUM_ENV = 1
    USE_SUBPROC = False
else:
    GYM_NUM_ENV = 20
    USE_SUBPROC = True

RENDER_HUMAN = 'human'
RENDER_RGB_ARRAY = 'rgb_array'

EVAL_METRIC_REWARD = 'reward'
EVAL_METRIC_F1 = 'f1'
EVAL_METRIC_COVERAGE_PROP = 'coverage_prop'
EVAL_METRIC_INTENSITY_PROP = 'intensity_prop'
EVAL_METRIC_MS1_MS2_RATIO = 'ms1ms2_ratio'
EVAL_METRIC_EFFICIENCY = 'efficiency'


def clip_value(value, max_value, min_range=0.0, max_range=1.0):
    '''
    Scale value by max_value, then clip it to [-min_range, max_range]
    '''
    value = min(value, max_value) if value >= 0 else max(value, -max_value)
    value = value / max_value
    if value < min_range:
        value = min_range
    elif value > max_range:
        value = max_range
    return value


def scale_intensity(original_intensity, log=False):
    if log:
        scaled_intensity = clip_value(np.log(original_intensity), MAX_OBSERVED_LOG_INTENSITY)
    else:
        scaled_intensity = clip_value(original_intensity, MAX_OBSERVED_INTENSITY)
    return scaled_intensity


def render_scan(scan):
    if scan is None:
        return None

    fig = plt.figure()
    for i in range(scan.num_peaks):
        x1 = scan.mzs[i]
        x2 = scan.mzs[i]
        y1 = 0
        y2 = np.log(scan.intensities[i])
        a = [[x1, y1], [x2, y2]]
        plt.plot(*zip(*a), marker='', color='r', ls='-', lw=1)
    plt.title('Scan {0} {1}s -- {2} peaks (ms_level {3})'.format(scan.scan_id, scan.rt,
                                                                 scan.num_peaks,
                                                                 scan.ms_level))
    plt.xlabel('m/z')
    plt.ylabel('log intensity')
    return fig


def linear_schedule(initial_value, min_value=0.0):
    def func(progress_remaining):
        return max(progress_remaining * initial_value, min_value)

    return func


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def evaluate(env, intensity_threshold=EVAL_F1_INTENSITY_THRESHOLD, format_output=True):
    # env can be either a DDAEnv or a ViMMS' Environment object
    try:
        vimms_env = env.vimms_env
    except AttributeError:
        vimms_env = env

    # call vimms codes to compute various statistics
    vimms_env_res = evaluate_simulated_env(vimms_env)
    count_fragmented = np.count_nonzero(vimms_env_res['times_fragmented'])
    count_ms1 = len(vimms_env.controller.scans[1])
    count_ms2 = len(vimms_env.controller.scans[2])
    try:
        ms1_ms2_ratio = float(count_ms1) / count_ms2
    except ZeroDivisionError:
        ms1_ms2_ratio = 0.0
    try:
        efficiency = float(count_fragmented) / count_ms2
    except ZeroDivisionError:
        efficiency = 0.0

    # get all base chemicals used as input to the mass spec
    all_chems = set(
        chem.get_original_parent() for chem in vimms_env.mass_spec.chemicals
    )

    # assume all base chemicals are unfragmented
    fragmented_intensities = {chem: 0.0 for chem in all_chems}

    # loop through ms2 scans, getting frag_events
    for ms2_scan in vimms_env.controller.scans[2]:
        frag_events = ms2_scan.fragevent
        if frag_events is not None:  # if a chemical has been fragmented ...

            # get the frag events for this scan
            # there would be one frag event for each chemical fragmented in this MS2 scan
            for event in frag_events:

                # get the base chemical that was fragmented
                base_chem = event.chem.get_original_parent()

                # store the max intensity of fragmentation for this base chem
                parent_intensity = event.parents_intensity[0]
                fragmented_intensities[base_chem] = max(
                    parent_intensity, fragmented_intensities[base_chem])

    TP = 0  # chemicals hit correctly (above threshold)
    FP = 0  # chemicals hit incorrectly (below threshold)
    FN = 0  # chemicals not hit
    total_frag_intensities = []
    for chem in fragmented_intensities:
        frag_int = fragmented_intensities[chem]
        max_intensity = chem.max_intensity
        if frag_int > 0:  # chemical was fragmented ...
            if fragmented_intensities[chem] > (intensity_threshold * max_intensity):
                TP += 1  # above threshold
            else:
                FP += 1  # below threshold
        else:
            FN += 1  # chemical was not fragmented
        total_frag_intensities.append(frag_int/max_intensity)

    assert (TP+FP+FN) == len(all_chems)
    assert len(total_frag_intensities) == len(all_chems)

    # ensure that coverage proportion calculation is consistent with ViMMS
    coverage_prop = vimms_env_res['coverage_proportion'][0]
    recalculated_coverage_prop = (TP+FP)/(TP+FP+FN)
    assert coverage_prop == recalculated_coverage_prop, \
        'coverage_prop %f is not the same as recalculated_coverage_prop %f' % (
            coverage_prop, recalculated_coverage_prop)

    # ensure that intensity proportion calculation is consistent with ViMMS
    intensity_prop = vimms_env_res['intensity_proportion'][0]
    recalculated_intensity_prop = np.mean(total_frag_intensities)
    assert intensity_prop == recalculated_intensity_prop
    assert intensity_prop == recalculated_intensity_prop, \
        'intensity_prop %f is not the same as recalculated_intensity_prop %f' % (
            intensity_prop, recalculated_intensity_prop)

    # compute precision, recall, f1
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        precision = 0.0

    try:
        f1 = 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0.0

    if format_output:
        eval_res = {
            'coverage_prop': '%.3f' % coverage_prop,
            'intensity_prop': '%.3f' % intensity_prop,
            'ms1ms2_ratio': '%.3f' % ms1_ms2_ratio,
            'efficiency': '%.3f' % efficiency,
            'TP': '%d' % TP,
            'FP': '%d' % FP,
            'FN': '%d' % FN,
            'precision': '%.3f' % precision,
            'recall': '%.3f' % recall,
            'f1': '%.3f' % f1
        }
    else:
        eval_res = {
            'coverage_prop': coverage_prop,
            'intensity_prop': intensity_prop,
            'ms1/ms1ms2_ratio': ms1_ms2_ratio,
            'efficiency': efficiency,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return eval_res
