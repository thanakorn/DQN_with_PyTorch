import numpy as np
import random
from policy.base_policy import BasePolicy

class EpsilonGreedy(BasePolicy):
    def __init__(self, init_epsilon, min_epsilon, decay):
        super().__init__()
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.min_epsilon + (self.init_epsilon - self.min_epsilon) * np.exp(-1. * current_step * self.decay)
        
    def choose_action(self, values, current_step):
        if np.random.rand() < self.get_exploration_rate(current_step):
            random.choice(range(len(values)))
        else:
            return np.argmax(values)
        