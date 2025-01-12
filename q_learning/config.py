import os
import yaml

from typing import Dict


PARAMS_FILE = os.path.join(os.path.dirname(__file__), 'params.yml')


class AgentParams:
    env_id: str
    learning_rate: float
    discount: float
    episodes: int
    q_table_dims: int
    max_punishment: int
    start_epsilon: float
    epsilon_decay: float
    min_epsilon: float

    def __init__(self, params_dict: Dict) -> None:
        for key in params_dict:
            setattr(self, key, params_dict[key])


def make_agent_params(config_name: str) -> AgentParams:
    with open(PARAMS_FILE, 'r') as f:
        all_params_dict = yaml.safe_load(f)

    assert config_name in all_params_dict, f'No config named {config_name} presented in {PARAMS_FILE}'
    params_dict = all_params_dict.get(config_name)

    return AgentParams(params_dict)
