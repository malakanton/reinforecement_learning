import os
import logging
import argparse
from types import NoneType
import torch
import time
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import itertools
import yaml
import matplotlib.pyplot as plt

from typing import Dict, List, Union

from dqn import QNet
from replay_memory import ReplayMemory


DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'


class AgentParams:
    env_id: str
    replay_memory_size: int
    batch_size: int
    epsilon_init: float
    epsilon_min: float
    epsilon_decay: float
    sync_rate: int
    learning_rate: float
    discount_factor: float

    def __init__(self, params_dict: Dict) -> None:
        for key in params_dict:
            setattr(self, key, params_dict[key])


def make_agent_params(params_dict: Dict) -> AgentParams:
    return AgentParams(params_dict)


class Agent:

    def __init__(self, config_name: str, dir_name: str = 'launches'):
        self.config_name = config_name
        self.dir_name = dir_name

        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)

        with open('DQN/params.yml') as f:
            all_params = yaml.safe_load(f)

        params = all_params.get(self.config_name )
        assert params is not None, f'No such paramenters set found: [{self.config_name }]'

        self.params = make_agent_params(params)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def optimize(self, batch: List, policy: QNet, target: QNet) -> None:

        prev_states, actions, new_states, rewards, terminations = zip(*batch)

        prev_states = torch.stack(prev_states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(DEVICE)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.params.discount_factor * target(new_states).max(dim=1)[0]


        current_q = policy(prev_states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    @staticmethod
    def _to_tensor(object: Union[int, float, np.ndarray]) -> torch.Tensor:
        if isinstance(object, (int, np.int64)):
            dtype = torch.int64
        elif isinstance(object, float):
            dtype = torch.float
        elif isinstance(object, np.ndarray):
            dtype = torch.float

        return torch.tensor(object, dtype=dtype, device=DEVICE)
    
    def _save_plot(self, rewards_per_episode, mean_window: int = 20) -> None:
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode) // mean_window)
        mean_rewards_idx = np.zeros(len(mean_rewards))

        for i in range(len(mean_rewards)):

            mean_rewards[i] = np.mean(rewards_per_episode[i * mean_window:(i + 1) * mean_window])
            mean_rewards_idx[i] = i * mean_window

        plt.plot(mean_rewards_idx, mean_rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Episodes')
        plt.title('Mean Rewards')
        
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        file_name = f'{self.config_name}_plot.png'

        fig.savefig(os.path.join(self.dir_name, file_name))
        plt.close(fig)

    def _save_model(self, model) -> None:
        file_name = f'{self.config_name}_policy.pt'
        model_path = os.path.join(self.dir_name, file_name)
        torch.save(model.state_dict(), model_path)

    def run(self, is_training=True, render=False) -> None:

        env = gym.make(self.params.env_id, render_mode="human" if render else None)

        states_num = env.observation_space.shape[0]
        actions_num = env.action_space.n

        rewards_per_episode = list()
        episodes_timings = list()
        epsilon_hist = list()
        best_reward = 0

        policy = QNet(states_num, actions_num).to(DEVICE)

        if is_training:
            memory = ReplayMemory(self.params.replay_memory_size)
            epsilon = self.params.epsilon_init
            target_net = QNet(states_num, actions_num).to(DEVICE)
            target_net.load_state_dict(policy.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.params.learning_rate)

        else:
            policy.load_state_dict(torch.load(os.path.join(self.dir_name, f'{self.config_name}_policy.pt')))
            policy.eval()

        optimised_times = 0

        for episode in itertools.count():

            prev_state, _ = env.reset()
            prev_state = self._to_tensor(object=prev_state)
            episode_reward = 0.0
            terminated = False
            
            ep_start = time.time()

            while not terminated:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = self._to_tensor(object=action)
                else:
                    with torch.no_grad():
                        action = policy(prev_state.unsqueeze(0)).squeeze().argmax()
                
                # print('action', action.item())
                new_state, reward, terminated, _, _ = env.step(action.item())

                reward = self._to_tensor(reward)
                new_state = self._to_tensor(object=new_state)

                episode_reward += reward

                if is_training:
                    memory.add((prev_state, action, new_state, reward, terminated))
                    step_count += 1

                    if len(memory.memory) > self.params.batch_size:
                        mini_batch = memory.sample(self.params.batch_size)

                        self.optimize(mini_batch, policy, target_net)
                        optimised_times += 1

                        if step_count > self.params.sync_rate:
                            target_net.load_state_dict(policy.state_dict())
                            step_count = 0
                
                    ep_finish = time.time()
                    ep_time = ep_finish - ep_start
                
                if episode_reward < -100:
                    break

                prev_state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:

                epsilon_hist.append(epsilon)
                epsilon = max(epsilon * self.params.epsilon_decay, self.params.epsilon_min)
                episodes_timings.append(ep_time)

                if episode_reward > best_reward:
                    self._save_model(policy)

                if episode % 100 == 0:
                    print(f'\nEpisode {episode}')
                    print(f'Training: {is_training}')
                    print(f'Epsilon: {epsilon}, Memory size {len(memory)}')
                    print(f'Time {ep_time} secs Reward: {episode_reward}, Best reward: {best_reward}')
                    if len(rewards_per_episode) > 0 and len(episodes_timings) > 0:
                        print(f'Mean episode reward {sum(rewards_per_episode) / len(rewards_per_episode)}')
                        print(f'Mean episode Time {sum(episodes_timings) / len(episodes_timings)}')

                    self._save_plot(rewards_per_episode)
            else:
                print(f'Episode {episode} Reward {episode_reward}')

            if episode_reward > best_reward:
                best_reward = episode_reward

        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('--config', '-c', type=str, required=True, help='Name of the config')
    parser.add_argument('--train', help='Train or test', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    render = False if args.train else True

    agent = Agent(args.config)
    agent.run(is_training=args.train, render=render)
