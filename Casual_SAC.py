from util import test_model,asymmetric_l2_loss,compute_mmd
from collections import namedtuple, deque
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from Agent.actor import DiagGaussianActor
from Agent.critic import DoubleQCritic
from Casual_Feature import CausalFeatureExtractor
from concurrent.futures import ThreadPoolExecutor
import os
import json
from datetime import datetime

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done', 'state_index', 'next_state_index'))
warnings.filterwarnings("ignore", category=UserWarning)


class SACAgent:
    def __init__(self, state_dim, action_dim, env, test_dataset, device,actor_lr=0.0003, critic_lr=0.0003,
                 gamma=0.99, tau=0.2, warmup_steps=5000, step_per_episode=5000, max_memory_size=100000, target_entropy=-1,gat_dim=128,
                 alpha_lr=3e-4, seed=42,mlp_hidden_dim=[32,20]):
        self.action_dim = action_dim
        self.max_memory_size = max_memory_size
        self.step_per_episode = step_per_episode
        self.device = device
        self.test_dataset = test_dataset
        self.warmup_steps = warmup_steps
        self.env = env
        self.gat_dim=gat_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        extractor_mlp_dim = [32,32,state_dim]
        self.casual_extractor = CausalFeatureExtractor\
            (input_dim=state_dim,gat_output_dim=self.gat_dim,mlp_hidden_dim=extractor_mlp_dim,device=device).to(self.device)

        self.actor = DiagGaussianActor(obs_dim=state_dim, action_dim=1, hidden_dim=self.mlp_hidden_dim,
                                       log_std_bounds=[-5, 2],device=device).to(self.device)
        self.Qnet = DoubleQCritic(obs_dim=state_dim, action_dim=1, hidden_dim=self.mlp_hidden_dim).to(self.device)
        self.TargetQnet = DoubleQCritic(obs_dim=state_dim, action_dim=1, hidden_dim=self.mlp_hidden_dim).to(
            self.device)

        self.env.agent = self.actor

        self.TargetQnet.load_state_dict(self.Qnet.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr,
                                          weight_decay=1e-4
                                          )

        self.Qnet_optimizer = optim.Adam(self.Qnet.parameters(), lr=critic_lr
                                         )

        self.casual_extractor_optimizer = optim.RMSprop(
            self.casual_extractor.parameters(),
            lr = 0.0003,
            momentum=0.95,
            eps=0.01,
            weight_decay=0.001
        )
        self.executor = ThreadPoolExecutor(max_workers=16)
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.init_params = {
            'model_name':'casual_model',
            'state_dim': state_dim,
            'action_dim': action_dim,
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'gamma': gamma,
            'tau': tau,
            'warmup_steps': warmup_steps,
            'step_per_episode': step_per_episode,
            'max_memory_size': max_memory_size,
            'target_entropy': target_entropy,
            'alpha_lr': alpha_lr,
            'seed': seed,
            'mlp_hidden_dim': mlp_hidden_dim,
            'gat_dim':self.gat_dim,
            'mlp_hidden_dim_casual_extractor': extractor_mlp_dim,
            'mlp_hidden_dim_actor': mlp_hidden_dim,
            'mlp_hidden_dim_Qnet': mlp_hidden_dim,
            'mlp_hidden_dim_TargetQnet': mlp_hidden_dim,
            'Env.TH':self.env.TH,
            'Env.TC':self.env.TC,
            'TH_update_interval':self.env.TH_update_interval,
            'deque_len':self.env.deque_len
        }
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def save_init_params(self, save_dir):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Training start at {}".format(self.timestamp))
        self.filename = f"params_{self.timestamp}.json"
        self.save_path = os.path.join(save_dir, self.filename)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(self.save_path, 'w') as f:
            json.dump(self.init_params, f, indent=4)
    def update_critic(self, obs, action, reward, next_obs, not_done):
        batch_size = obs.shape[0]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                next_obs)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_obs
                                           if s is not None]).view(batch_size, -1)

        with torch.no_grad():
            target_Q = torch.zeros_like(reward)

            dist = self.actor(non_final_next_states)
            next_action = self.actor.sample_action(non_final_next_states)
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.TargetQnet(non_final_next_states, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.log_alpha * log_prob

            target_Q[non_final_mask] = reward[non_final_mask] +\
                                       (not_done[non_final_mask] * self.gamma * target_V[non_final_mask])

        current_Q1, current_Q2 = self.Qnet(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.Qnet_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qnet.parameters(),1.0)
        self.Qnet_optimizer.step()

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = self.actor.sample_action(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.Qnet(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = asymmetric_l2_loss((self.log_alpha * log_prob-actor_Q),tau=0.5)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.log_alpha.exp() *
                      (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_casual_extractor(self, obs):

        basic_feature = self.casual_extractor(obs, use_casual=True)
        origin_feature = self.casual_extractor(obs, use_casual=False)

        with torch.no_grad():
            obs_action = self.actor.sample_action(obs,deterministic=False)
        rec_action = self.casual_extractor.get_pseudo_label(basic_feature)
        losses = asymmetric_l2_loss((obs_action - rec_action), tau=0.5)
        feature_loss = compute_mmd(basic_feature, origin_feature)
        loss = feature_loss + losses

        self.casual_extractor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.casual_extractor.parameters(), max_norm=1)
        self.casual_extractor_optimizer.step()

    def update(self, batch, step):
        states = torch.cat(batch.state).to(dtype=torch.float32, device=self.device)
        actions = torch.cat(batch.action).to(dtype=torch.long, device=self.device)
        rewards = torch.cat(batch.reward).unsqueeze(-1).to(dtype=torch.float32, device=self.device)
        next_states = torch.cat(batch.next_state).to(dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(-1).to(device=self.device)
        c_states = states
        c_next_states = next_states

        self.update_critic(c_states, actions, rewards, c_next_states, dones)
        self.update_actor_and_alpha(c_states)

        if step % 2 == 0:
            self.soft_update(self.TargetQnet, self.Qnet)
            self.update_casual_extractor(states)

    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def warmup(self):
        self.is_warmup = True
        env_states = torch.tensor(self.env.x, dtype=torch.float32, device=self.device)
        state_idx = self.env.reset()
        state = torch.tensor(self.env.x[state_idx, :], dtype=torch.float32, device=self.device).unsqueeze(0)
        for _ in tqdm(range(self.warmup_steps), desc='Warm Steps', leave=False):
            with torch.no_grad():
                casual_state = self.casual_extractor(state, use_casual=True)

            action = self.actor.sample_action(casual_state, deterministic=False)
            future = self.executor.submit(self.env.step, action.item(), self.is_warmup)
            observation_i, rewards, done, _ = future.result()
            observation = env_states[observation_i].unsqueeze(0)

            self.memory.add(state, action,
                            observation, torch.tensor([rewards], dtype=torch.float32, device=self.device),
                            done, state_idx, observation_i)

            state = observation
            state_idx = observation_i

            self.update_casual_extractor(state)
        return

    def fit(self):
        self.save_init_params('./log/')
        self.env.fit_IForest()
        self.reset_counters()
        self.reset_memory()
        self.warmup()
        self.is_warmup = False
        env_states = torch.tensor(self.env.x, dtype=torch.float32, device=self.device)
        for e in range(10):
            reward_history = []
            state_idx = self.env.reset()
            state = torch.tensor(self.env.x[state_idx, :], dtype=torch.float32, device=self.device).unsqueeze(0)

            for step in tqdm(range(self.step_per_episode), desc='Training Steps', leave=False):

                self.num_steps_done += 1
                with torch.no_grad():
                    Cstate= self.casual_extractor(state,use_casual=True)
                action = self.actor.sample_action(Cstate, deterministic=False)
                future = self.executor.submit(self.env.step, action.item(), self.is_warmup)
                observation_i, rewards, done, _ = future.result()
                reward_history.append(rewards)
                self.wandbrun.log({"action":action.item(),"reward":rewards,"step":self.num_steps_done,"Env_TH": self.env.TH})

                observation = env_states[observation_i].unsqueeze(0)

                self.memory.add(state, action,
                                observation, torch.tensor([rewards], dtype=torch.float32, device=self.device),
                                done, state_idx, observation_i)

                state = observation
                state_idx = observation_i

                if self.num_steps_done % 16 ==0 and self.num_steps_done >= self.warmup_steps:
                    for i in range(16):
                        transitions = self.memory.sample(64)
                        batch = Transition(*zip(*transitions))
                        self.update(batch,self.num_steps_done)


                if self.num_steps_done % 100 == 0:
                    auc, pr = test_model(self.test_dataset, self.actor, self.device)
                    self.wandbrun.log({"auc":auc,"pr":pr})
                    self.pr_auc_history.append(pr)
                    self.roc_auc_history.append(auc)


            avg_reward = np.mean(reward_history)
            message = 'Episode: {} \t Steps: {} \t Average episode Reward: {} \t Best Auc: {}'.format(e, step + 1,avg_reward,max(self.roc_auc_history))
            print(message)
        self.executor.shutdown(wait=True)

    def show_results(self):

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].plot(self.episodes_total_reward)
        axs[0].set_title('Total reward per episode')
        axs[1].plot(self.pr_auc_history)
        axs[1].set_title('PR AUC per validation step')
        axs[2].plot(self.roc_auc_history)
        axs[2].set_title('ROC AUC per validation step')
        plt.show()

    def model_performance(self):
        """
        Test the model
        :param on_test_set: whether to test on the test set or the validation set
        """
        print(max(self.roc_auc_history))
        if max(self.roc_auc_history)>=0.75:
            save_dir = './log/roc_auc_history/'
            filename = f"history_{self.timestamp}.json"
            save_path = os.path.join(save_dir, filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_path,'w') as f:
                json.dump(self.roc_auc_history,f,indent=4)

        with open(self.save_path, 'a') as f:
            json.dump(max(self.roc_auc_history), f, indent=4)
        return test_model(self.test_dataset, self.actor, self.device)

    def reset_memory(self):
        self.memory = ReplayBuffer(self.max_memory_size)

    def reset_counters(self):
        # training counters and utils
        self.num_steps_done = 0
        self.episodes_total_reward = []
        self.pr_auc_history = []
        self.roc_auc_history = []
        self.best_pr = None

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
