class BasePolicy(object):
    def __init__(self):
        super().__init__()
        
    def choose_action(self, values, current_step):
        raise NotImplementedError()