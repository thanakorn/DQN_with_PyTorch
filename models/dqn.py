import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDQN(nn.Module):
    
    def __init__(self, img_height, img_width, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_features=img_height * img_width * 3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=num_actions)
        
    def forward(self, s):
        s = s.flatten(start_dim=1)
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(out))
        out = self.out(out)
        return out
        