import matplotlib.pyplot as plt
import matplotlib
import torch

from IPython import display

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episodes')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_avg(moving_avg_period, values))
    plt.pause(0.001)
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython: display.clear_output(wait=True)
    
def get_moving_avg(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()