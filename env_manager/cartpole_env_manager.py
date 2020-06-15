import gym
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image

class CartPoleEnvManager(object):
    def __init__(self, device):
        super().__init__()
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.device = device
        
    def reset(self):
        self.env.reset()
        self.current_screen = None
        self.done = False
        
    def close(self):
        self.env.close()
        
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def num_actions(self):
        return self.env.action_space.n
    
    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action)
        return torch.tensor([reward], device=self.device)
    
    def just_start(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_start() or self.done:
            self.current_screen = self.get_processed_screen()
            empty_screen = torch.zeros_like(self.current_screen)
            return empty_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
        
    def get_img_dimensions(self):
        return self.get_processed_screen().shape
        
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen(screen)
    
    def crop_screen(self, screen):
        height = screen.shape[1]
        top = int(height * 0.4)
        bottom = int(height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen
    
    def transform_screen(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
        screen = torch.from_numpy(screen)
        
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40, 90)),
            T.ToTensor()
        ])
        
        return resize(screen).unsqueeze(0).to(self.device)