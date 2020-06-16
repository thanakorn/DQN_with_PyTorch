import torch
from memory.replay_memory import Experience

def extract_experiences(experiences):
    batch = Experience(*zip(*experiences))
    
    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)
    
    return (states, actions, rewards, next_states)

def get_q_next(target_net, next_states):
    final_idx = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    non_final_idx = (final_idx == False)
    non_final_states = next_states[non_final_idx]
    values = torch.zeros(next_states.shape[0])
    values[non_final_idx] = target_net(non_final_states).max(dim=1)[0]
    return values