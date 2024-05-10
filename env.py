import gym
import time
import numpy as np
from util import get_latent, compute_avg_similarity
from gym import spaces
import torch
from sklearn.ensemble import IsolationForest
from collections import deque


class ADEnv(gym.Env):

    def __init__(self, dataset: np.ndarray, sampling_Du=1000, prob_au=0.5, label_normal=0, label_anomaly=1,
                 name="default", TC=6, TH=0.1, device='cuda', deque_len=50, TH_update_interval=100):

        super().__init__()
        self.randomness = 0.3
        self.name = name
        self.agent = None
        dataset = torch.tensor(dataset).to(device)

        self.num_S = sampling_Du
        self.normal = label_normal
        self.anomaly = label_anomaly
        self.prob = prob_au
        self.deque_len = deque_len
        self.m, self.n = dataset.shape

        self.n_feature = self.n - 1
        self.n_samples = self.m
        self.x = dataset[:, :self.n_feature]
        self.y = dataset[:, self.n_feature]
        self.dataset = dataset
        self.index_u = (self.y == self.normal).nonzero(as_tuple=True)[0]
        self.index_a = (self.y == self.anomaly).nonzero(as_tuple=True)[0]

        self.TC = TC
        self.TH = TH

        self.TH_history = []
        self.action_history = []
        self.TH_update_interval = TH_update_interval
        self.state_history = deque(maxlen=deque_len)
        self.Temporary_data = {}
        self.observation_space = spaces.Discrete(self.m)
        self.action_space = spaces.Discrete(2)

        self.counts = None
        self.state = None

        self.normalize_scores = None

    def Update_TH(self):
        action_history_tensor = torch.tensor(self.action_history[-self.TH_update_interval:], device=self.agent.device)
        current_anomaly_ratio = torch.mean((action_history_tensor >= self.TH).float()).item()

        target_anomaly_ratio = 0.1

        ratio_difference = current_anomaly_ratio - target_anomaly_ratio

        adjustment_factor = 1.0 + (ratio_difference * 0.1)

        adjustment_factor = max(0.9, min(1.1, adjustment_factor))


        self.TH *= adjustment_factor

        self.TH = max(0.05, min(self.TH, 1))

    def fit_IForest(self):
        print("Constructing IForest")
        all_state = self.agent.trunk(torch.tensor(self.x, dtype=torch.float32).to(self.agent.device))
        all_state_cpu = all_state.cpu().detach().numpy()

        self.iforest = IsolationForest(contamination='auto')
        self.iforest.fit(all_state_cpu)
        scores = self.iforest.decision_function(all_state_cpu)
        self.normalize_scores = [-1 * s + 0.5 for s in scores]

    def generate_a(self, *args, **kwargs):
        index = torch.randint(len(self.index_a), (1,)).item()
        return index

    def generate_t(self, *args, **kwargs):
        s_ts = torch.tensor(list(self.Temporary_data.keys()), device=self.agent.device)
        index = s_ts[torch.randint(len(s_ts), (1,))].item()
        return index

    def generate_u(self, action, s_t):

        if len(self.index_u) == 0:
            all_states = torch.cat([torch.tensor(self.index_a, device=self.agent.device),
                                    torch.tensor(list(self.Temporary_data.keys()), device=self.agent.device)])
            return torch.randint(0, len(all_states), (1,)).item()
        else:

            S = torch.randint(0, len(self.index_u), (self.num_S,)).to(self.agent.device)

        all_x = self.x[torch.cat([S, torch.tensor([s_t], device=self.agent.device)])]
        all_x_tensor = all_x.float().to(self.agent.device)

        all_state = all_x_tensor
        history_state = self.x[self.state_history]
        avg_similarities = torch.tensor([compute_avg_similarity(point, history_state) for point in all_state])

        random_scores = torch.rand_like(avg_similarities)
        combined_scores = avg_similarities * (1 - self.randomness) + random_scores * self.randomness

        selected_index = torch.argmax(combined_scores).item()
        return selected_index

    def update_data_pools(self, s_t, action):
        if s_t in self.index_a:
            return
        else:
            if action >= self.TH:
                if s_t in self.index_u and s_t not in self.Temporary_data:
                    self.Temporary_data[s_t] = 1
                    self.index_u = self.index_u[self.index_u != s_t]
                elif s_t in self.Temporary_data and self.Temporary_data[s_t] < self.TC:
                    self.Temporary_data[s_t] += 1
                elif s_t in self.Temporary_data and self.Temporary_data[s_t] >= self.TC:
                    self.index_a = torch.cat([self.index_a, torch.tensor([s_t], device=self.agent.device)])
                    del self.Temporary_data[s_t]
                else:
                    return

            else:
                if s_t in self.index_u:
                    return
                elif s_t in self.Temporary_data and (0 < self.Temporary_data[s_t] < self.TC):
                    self.Temporary_data[s_t] -= 1
                elif s_t in self.Temporary_data and self.Temporary_data[s_t] == 0:
                    self.index_u = torch.cat([self.index_u, torch.tensor([s_t], device=self.agent.device)])
                    del self.Temporary_data[s_t]
                else:
                    return

    def combined_reward(self, action, s_t):
        if s_t in self.index_a:
            reward = max(0, action - self.TH) * 2 + 1.0
        elif s_t in self.Temporary_data.keys():
            temp_data_count = self.Temporary_data[s_t]
            reward = (temp_data_count / self.TC) * 1 if action >= self.TH else -1
        elif s_t in self.index_u:
            normalized_score = self.normalize_scores[s_t]
            reward = (action - self.TH) * normalized_score
        else:
            reward = -0.01
        return reward

    def step(self, action, is_warmup):

        self.state = int(self.state)
        self.action_history.append(action)
        self.state_history.append(self.state)
        s_t = self.state
        if len(self.Temporary_data.values()) == 0:
            g = np.random.choice([self.generate_a, self.generate_u], p=[0.5, 0.5])
        elif len(self.index_u) == 0:
            g = np.random.choice([self.generate_a, self.generate_t], p=[0.5, 0.5])
        else:
            g = np.random.choice([self.generate_a, self.generate_t, self.generate_u], p=[0.3, 0.3, 0.4])
        s_tp1 = g(action, s_t)

        self.state = s_tp1
        self.state = int(self.state)
        self.counts += 1
        if len(self.action_history) % self.TH_update_interval == 0 and not is_warmup:
            self.Update_TH()
        reward = self.combined_reward(action, s_t)
        if not is_warmup:
            self.update_data_pools(s_t, action)
        done = False

        info = {"State t": s_t, "Action t": action, "State t+1": s_tp1}

        return self.state, reward, done, info

    def reset(self):
        self.counts = 0
        if len(self.index_u) != 0:
            self.state = self.index_u[torch.randint(len(self.index_u), (1,))].item()
        else:
            self.state = self.generate_t()

        return self.state