import numpy as np
from vimms.Evaluation import EvaluationData

from vimms_gym.common import EVAL_F1_INTENSITY_THRESHOLD, HISTORY_HORIZON, METHOD_RANDOM, \
    METHOD_FULLSCAN, METHOD_TOPN, METHOD_PPO, METHOD_PPO_RECURRENT, METHOD_DQN, evaluate
from vimms_gym.env import DDAEnv
from vimms_gym.policy import random_policy, fullscan_policy, topN_policy, get_ppo_action_probs, \
    get_dqn_q_values
from vimms_gym.wrappers import HistoryWrapper, flatten_dict_observations


class Episode():
    def __init__(self, initial_obs):
        self.env = None
        self.rewards = []
        self.observations = [initial_obs]
        self.actions = []
        self.action_probs = []
        self.infos = []
        self.num_steps = 0
        self.scans = None

    def add_step_data(self, action, action_probs, obs, reward, info):
        self.actions.append(action)
        self.action_probs.append(action_probs)
        self.observations.append(obs)
        self.rewards.append(reward)
        self.infos.append(info)
        self.num_steps += 1

    def get_step_data(self, i):
        return {
            'state': self.observations[i],
            'reward': self.rewards[i],
            'action': self.actions[i],
            'action_prob': self.action_probs[i],
            'info': self.infos[i]
        }

    def evaluate_environment(self, env, intensity_threshold):
        vimms_env = env.vimms_env
        self.eval_data = EvaluationData(vimms_env)
        self.eval_res = evaluate(vimms_env, intensity_threshold)
        self.eval_res['total_rewards'] = sum(self.rewards)
        self.eval_res['invalid_action_count'] = env.invalid_action_count
        return self.eval_res

    def get_total_rewards(self):
        return np.sum(self.rewards)


def run_method(env_name, env_params, max_peaks, chem_list, method, out_dir,
               N=10, min_ms1_intensity=5000, model=None,
               print_eval=False, print_reward=False, mzml_prefix=None,
               intensity_threshold=EVAL_F1_INTENSITY_THRESHOLD):

    if METHOD_DQN in method:
        assert model is not None
    if METHOD_PPO in method:
        assert model is not None

    # to store all results across all loop of chem_list
    all_episodic_results = []

    for i in range(len(chem_list)):
        chems = chem_list[i]
        if print_reward:
            print(f'\nEpisode {i} ({len(chems)} chemicals)')

        env = DDAEnv(max_peaks, env_params)
        env = flatten_dict_observations(env)
        env = HistoryWrapper(env, horizon=HISTORY_HORIZON)    
        obs = env.reset(chems=chems)
        states = None
        done = False

        # lists to store episodic results
        episode = Episode(obs)
        episode_starts = np.ones((1,), dtype=bool)        
        while not done:  # repeat until episode is done

            unwrapped_obs = env.env.env.state # access the state attribute in DDAEnv

            # select an action depending on the observation and method
            action, action_probs, states = pick_action(
                method, obs, unwrapped_obs, model, env.features, N, min_ms1_intensity, states=states, episode_starts=episode_starts)

            # make one step through the simulation
            obs, reward, done, info = env.step(action)
            episode_starts = done
            
            # store new episodic information
            if obs is not None:
                episode.add_step_data(action, action_probs, obs, reward, info)

            if print_reward and episode.num_steps % 500 == 0:
                print('steps\t', episode.num_steps, '\ttotal rewards\t', episode.get_total_rewards())

            # if episode is finished, break
            if done:
                break

        if print_reward:
            print(
                f'Finished after {episode.num_steps} timesteps with '
                f'total reward {episode.get_total_rewards()}')

        # save mzML and other info useful for evaluation of the ViMMS environment
        mzml_name = mzml_prefix if not None else method
        out_file = '%s_%d.mzML' % (mzml_name, i)
        env.write_mzML(out_dir, out_file)

        # environment will be evaluated here
        eval_res = episode.evaluate_environment(env, intensity_threshold)
        if print_eval:
            print(eval_res)
        all_episodic_results.append(episode)

    return all_episodic_results


def pick_action(method, obs, unwrapped_obs, model, features, N, min_ms1_intensity, 
    states=None, episode_starts=None):
    action_probs = []

    if method != METHOD_PPO_RECURRENT:
        if METHOD_DQN in method:
            method = METHOD_DQN
        elif METHOD_PPO in method:
            method = METHOD_PPO

    if method == METHOD_RANDOM:
        action = random_policy(unwrapped_obs)
    elif method == METHOD_FULLSCAN:
        action = fullscan_policy(unwrapped_obs)
    elif method == METHOD_TOPN:
        action = topN_policy(unwrapped_obs, features, N, min_ms1_intensity)
    elif method == METHOD_PPO:
        action, states = model.predict(obs, deterministic=True)
        action_probs = get_ppo_action_probs(model, obs)
    elif method == METHOD_PPO_RECURRENT:
        action, states = model.predict(obs, deterministic=True, state=states, episode_start=episode_starts)
        # FIXME: this is not working yet
        # action_probs = get_recurrent_ppo_action_probs(model, obs, states, episode_starts)        
    elif method == METHOD_DQN:
        action, states = model.predict(obs, deterministic=True)
        q_values = get_dqn_q_values(model, obs)
        action_probs = q_values # not really ....

    return action, action_probs, states
