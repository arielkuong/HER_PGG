import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, size_in_transitions, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size_in_rollouts = size_in_transitions // self.T
        # memory management
        self.current_size_in_rollouts = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size_in_rollouts, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size_in_rollouts, self.T, self.env_params['action']]),
                        'd': np.zeros([self.size_in_rollouts, self.T + 1]), # distance between achieved goal and desired goal
                        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_d, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.buffers['d'][idxs] = mb_d
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            assert self.current_size_in_rollouts > 0
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size_in_rollouts]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size_in_rollouts, "Batch committed to replay is too large!"

        # Increment consecutively until hit the end
        if self.current_size_in_rollouts+inc <= self.size_in_rollouts:
            idx = np.arange(self.current_size_in_rollouts, self.current_size_in_rollouts+inc)
        elif self.current_size_in_rollouts < self.size_in_rollouts:
            overflow = inc - (self.size_in_rollouts - self.current_size_in_rollouts)
            idx_a = np.arange(self.current_size_in_rollouts, self.size_in_rollouts)
            idx_b = np.random.randint(0, self.current_size_in_rollouts, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size_in_rollouts, inc)

        # Update the replay size
        self.current_size_in_rollouts = min(self.size_in_rollouts, self.current_size_in_rollouts+inc)

        if inc == 1:
            idx = idx[0]
        return idx

    def goal_generation(self, stretagy):
        if stretagy == 'all':
            distances_all = self.buffers['d'][:self.current_size_in_rollouts].copy()
            distances_all = distances_all.flatten()
            index = np.argmin(distances_all)
            selected_goal = self.buffers['ag'][int(index/(self.T + 1)), index%(self.T + 1)].copy()
            index_distance = self.buffers['d'][int(index/(self.T+1)), index%(self.T + 1)].copy()
            print('generated goal selected from episode {}, timestep {}, distance to original goal: {}'.format(int(index/(self.T+1)), index%(self.T + 1), index_distance))
        elif stretagy == 'final':
            distances_final = self.buffers['d'][:self.current_size_in_rollouts, -1].copy()
            print(distances_final.shape)
            distances_final = distances_final.flatten()
            index = np.argmin(distances_final)
            print(index)
            selected_goal = self.buffers['ag'][index, self.T].copy()
            index_distance = self.buffers['d'][index, self.T].copy()
            print('generated goal selected from episode {}, timestep {}, distance to original goal: {}'.format(index, self.T, index_distance))
        else:
             assert stretagy == 'all' or stretagy == 'final', 'PGG stretagy unsupport!'

        return selected_goal
