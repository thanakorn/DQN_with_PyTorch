import random
from collections import namedtuple

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory(object):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
        
    def sample(self, batch_size):
        if len(self.memory) >= batch_size:
            return random.sample(self.memory, batch_size)
        else:
            raise Exception('Insufficient replay memory')