import gymnasium as gym
import numpy as np
import argparse

from typing import Tuple

from config import make_agent_params, AgentParams


class Agent:

    def __init__(self, params: AgentParams, train: bool = False):
        self.params = params
        self.env = gym.make(self.params.env_id, render_mode="human" if not train else None)
        self._init_q_table()
    
    def _init_q_table(self):
        discrete_os_size = [self.params.q_table_dims] * len(self.env.observation_space.high)
        self.discrete_os_win_size = (
            (self.env.observation_space.high - self.env.observation_space.low) / discrete_os_size
        )
        self.q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [self.env.action_space.n]))

    def _discrete_state(self, state: np.ndarray) -> Tuple[int, int]:
        discr_state = (state - self.env.observation_space.low) / self.discrete_os_win_size
        return tuple(discr_state.astype(np.int32))
    
    def _get_action(self, prev_state: np.ndarray) -> np.int32:
        descrete_state = self._discrete_state(prev_state)
        return np.argmax(self.q_table[descrete_state])
    
    def _calc_new_q_value(self, curr_q: float, max_next_q: float, reward: float) -> float:
        return (1 - self.params.learning_rate) * curr_q + \
              self.params.learning_rate * (reward + self.params.discount * max_next_q)
    
    def _render_env(self, render: bool):
        render = 'human' if render else None
        self.env = gym.make(self.params.env_id, render_mode=render)

    def _update_q_table(self, prev_state: np.ndarray, new_state: np.ndarray, action: np.int32, reward: int) -> None:
        prev_descrete_state = self._discrete_state(prev_state)
        new_descrete_state = self._discrete_state(new_state)

        max_next_q_value = np.max(self.q_table[new_descrete_state])
        current_q_value = self.q_table[prev_descrete_state][action]

        new_q = self._calc_new_q_value(current_q_value, max_next_q_value,  reward)
        self.q_table[prev_descrete_state][action] = new_q

    def run(self, verbose: int) -> None:

        epsilon = self.params.start_epsilon

        for episode in range(1, self.params.episodes + 1):
            if episode % verbose == 0:
                print(f'Episode {episode}, epsilon {epsilon}')
                print(self.params.learning_rate)

            terminated, truncated = False, False
            self._render_env(render=episode % verbose == 0)
            prev_state, _ = self.env.reset()
            episode_reward = 0

            while not terminated and not truncated:
                if np.random.random() > epsilon:
                    action = self._get_action(prev_state)
                else:
                    action = self.env.action_space.sample()

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                
                if not terminated:
                    self._update_q_table(prev_state, new_state, action, reward)
                    
                prev_state = new_state
                episode_reward += reward

                if episode_reward < self.params.max_punishment:
                    break

            if terminated:
                print(f'EP {episode} ON THE HILL!!! Reward: {episode_reward}')
                self.params.learning_rate *= self.params.epsilon_decay
            
            epsilon = epsilon * self.params.epsilon_decay

            self.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('--config', '-c', type=str, required=True, help='Name of the config')
    parser.add_argument('--train', help='Train or test', action='store_true')
    parser.add_argument('--verbose', '-v', help='Frequency of episodes to render', type=int, default=500)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    params = make_agent_params(args.config)

    agent = Agent(params, args.train)

    agent.run(args.verbose)
