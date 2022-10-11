import pylab as plt
import seaborn as sns
import numpy as np


def plot_N(episodic_result, bins='auto', title=None):
    block_length = 0
    block_lengths = []

    for obs in episodic_result.observations:
        ms_level = obs['ms_level'][0]
        if ms_level == 1:
            block_lengths.append(block_length)
            block_length = 0
        elif ms_level == 2:
            block_length += 1

    sns.histplot(block_lengths, bins=bins)
    plt.xlabel('N')
    if title is not None:
        plt.title(title)


def plot_ms1_ms2_counts(episodic_result, title=None):
    ms_levels = np.array([int(obs['ms_level'][0]) for obs in episodic_result.observations])
    ms1_count = np.cumsum((ms_levels == 1))
    ms2_count = np.cumsum((ms_levels == 2))

    plt.plot(ms1_count, label='MS1')
    plt.plot(ms2_count, label='MS2')
    plt.ylabel('Cumulative counts')
    plt.xlabel('Step')
    plt.legend()
    if title is not None:
        plt.title(title)
    return ms1_count, ms2_count


def plot_action_hist(episodic_result, ms2_only=False, bins='auto', title=None):
    actions = episodic_result.actions
    if ms2_only:
        ms1_action_index = 200
        actions = [a for a in actions if a != ms1_action_index]
    sns.histplot(actions, bins=bins)
    plt.xlabel('Action')
    if title is not None:
        plt.title(title)


def get_selected_action_probs(episodic_result, limit, max_peaks):
    actions = episodic_result.actions
    action_probs = episodic_result.action_probs

    selected_actions = actions[0:limit]
    selected_action_probs = []
    all_selected_action_probs = []
    for i in range(len(selected_actions)):
        action = selected_actions[i]
        try:
            flattened_probs = action_probs[i].flatten()
        except AttributeError:  # TopN
            flattened_probs = [0.0] * (max_peaks + 1)
        all_selected_action_probs.append(flattened_probs)
        selected_action_probs.append(flattened_probs[action])

    all_selected_action_probs = np.array(selected_action_probs)
    return selected_actions, selected_action_probs, all_selected_action_probs


def plot_action_probs(episodic_result, limit, max_peaks, title=None):
    actions, action_probs, _ = get_selected_action_probs(episodic_result, limit, max_peaks)
    no_hue = False
    if min(action_probs) == 0 and max(action_probs) == 0:
        no_hue = True

    plt.figure(figsize=(20, 5))
    palette = sns.color_palette('icefire', as_cmap=True)
    if no_hue:
        sns.scatterplot(x=range(limit), y=actions, palette=palette)
    else:
        sns.scatterplot(x=range(limit), y=actions, hue=action_probs, palette=palette)
        plt.legend(title='action probability')

    plt.xlabel('Step')
    plt.ylabel('Action')
    if title is not None:
        plt.title(title)


def plot_reward_probs(episodic_result, limit, max_peaks, title=None):
    actions, action_probs, _ = get_selected_action_probs(episodic_result, limit, max_peaks)
    no_hue = False
    if min(action_probs) == 0 and max(action_probs) == 0:
        no_hue = True

    rewards = episodic_result.rewards[0:limit]

    plt.figure(figsize=(20, 5))
    palette = sns.color_palette('icefire', as_cmap=True)
    if no_hue:
        sns.scatterplot(x=range(limit), y=rewards, palette=palette)
    else:
        sns.scatterplot(x=range(limit), y=rewards, hue=action_probs, palette=palette)
        plt.legend(title='action probability')

    plt.xlabel('Step')
    plt.ylabel('Reward')
    if title is not None:
        plt.title(title)
