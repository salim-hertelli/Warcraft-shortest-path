from utils import *
from experiment import Experiment
from resnet18 import partialResNet
import pyepo
import torch
from torch import nn
from tqdm import tqdm

class SPOPlus(Experiment):
    def __init__(self, name, loader_train, loader_test, epochs, log_step, lr, optmodel, processes = 1):
        super().__init__(name, loader_train, loader_test, epochs, log_step, lr, optmodel)
        self._setup(processes)
    
    def _setup(self, processes):
        # init net
        self.nnet = partialResNet(k=12)
        # cuda
        if torch.cuda.is_available():
            self.nnet = self.nnet.cuda()
        # set optimizer
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=self.lr)
        # set stopper
        #stopper = earlyStopper(patience=7)
        # set loss
        self.spoploss = pyepo.func.SPOPlus(self.optmodel, processes=processes)
        # epochs = ?? FIXME
    
    def train(self):
        # train
        self.loss_log3, self.regret_log3 = [], [pyepo.metric.regret(self.nnet, self.optmodel, self.loader_test)]
        tbar = tqdm(range(self.epochs))
        for epoch in tbar:
            self.nnet.train()
            for x, c, w, z in self.loader_train:
                # cuda
                if torch.cuda.is_available():
                    x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                # forward pass
                cp = self.nnet(x) # predicted cost
                loss = self.spoploss(cp, c, w, z).mean() # loss
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # log loss
                self.loss_log3.append(loss.item())
                tbar.set_description("Epoch: {:2}, Loss: {:3.4f}".format(epoch, loss.item()))
            # scheduled learning rate
            if (epoch == int(self.epochs*0.6)) or (epoch == int(self.epochs*0.8)):
                for g in self.optimizer.param_groups:
                    g['lr'] /= 10
            if epoch % self.log_step == 0:
                # log regret
                regret = pyepo.metric.regret(self.nnet, self.optmodel, self.loader_test) # regret on test
                self.regret_log3.append(regret)
                # early stop
                #regret = pyepo.metric.regret(nnet, optmodel, loader_val) # regret on val
                #if stopper.stop(regret):
                #    break
        self.epoch = epoch
    
    def plot(self):
        # plot
        plotLearningCurve(self.loss_log3, self.regret_log3, self.epoch, self.epochs, self.log_step)
        # eval
        print("Test set:")
        df1 = evaluate(self.nnet, self.optmodel, self.loader_test)
        print("Train set:")
        df2 = evaluate(self.nnet, self.optmodel, self.loader_train)