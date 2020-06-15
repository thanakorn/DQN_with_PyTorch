import torch

class Agent(object):
    def __init__(self, policy, num_actions, policy_net, device):
        super().__init__()
        self.current_step = 0
        self.policy = policy
        self.num_actions = num_actions
        self.policy_net = policy_net
        self.device = device
        
    def act(self, state):
        self.current_step += 1
        with torch.no_grad():
            values = self.policy_net(state).to(self.device)
            return self.policy.choose_action(values, self.current_step)