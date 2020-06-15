#%%
import gym
import torch
import matplotlib
import matplotlib.pyplot as plt

from models.dqn import SimpleDQN
from memory.replay_memory import Experience, ReplayMemory
from policy.epsilon_greedy import EpsilonGreedy
from agent import Agent
from env_manager.cartpole_env_manager import CartPoleEnvManager

# %% Explore environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env_manager = CartPoleEnvManager(device)
env_manager.reset()
screen = env_manager.get_processed_screen()
env_manager.close()
plt.figure()
plt.title('Processes Screen')
plt.axis('off')
plt.imshow(screen.squeeze(0).permute(1,2,0))
plt.show()

screen = env_manager.get_state()
plt.figure()
plt.title('Initial State')
plt.axis('off')
plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
plt.show()

env_manager.take_action(1)
env_manager.take_action(1)
screen = env_manager.get_state()
plt.figure()
plt.title('State')
plt.axis('off')
plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
plt.show()
# %%
