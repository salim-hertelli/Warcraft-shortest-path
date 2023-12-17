# base class for experiments

class Experiment():
    def __init__(self, name, loader_train, loader_test, epochs, log_step, lr, optmodel):
        self.name = name
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.epochs = epochs
        self.log_step = log_step
        self.lr = lr
        self.optmodel = optmodel

    def get_name(self):
        return self.name
    
    def _setup(self):
        """
        A method to setup experiment
        """
        raise NotImplementedError
    
    def train(self):
        """
        A method to train experiment
        """
        raise NotImplementedError
    
    def plot(self):
        """
        A method to plot experiment
        """
        raise NotImplementedError