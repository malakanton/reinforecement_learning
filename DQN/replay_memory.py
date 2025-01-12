import random

from typing import Optional, Any
from collections import deque


class ReplayMemory:

    def __init__(self, max_len: int, seed: Optional[int] = None):
        self.memory = deque(maxlen=max_len)

        if seed is not None:
            random.seed(seed)

    def add(self, element: Any):
        self.memory.append(element)

    def sample(self, size: int):
        if size >= len(self.memory):
            size = len(self.memory)
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)
