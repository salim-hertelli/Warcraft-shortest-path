from utils import *
from experiment import Experiment
from resnet18 import partialResNet
import pyepo
import torch
from torch import nn
from tqdm import tqdm

class TwoPhaseBCE(Experiment):
    def __init__(self, name, loader_train, loader_test, epochs, log_step, lr, optmodel):
        super().__init__(name, loader_train, loader_test, epochs, log_step, lr, optmodel)
        self._setup()
    
    def _setup(self):
        # init net
        self.nnet = partialResNet(k=12)
        # cuda
        if torch.cuda.is_available():
            self.nnet = self.nnet.cuda()
        # set optimizer
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=self.lr)
        # set loss
        self.bceloss = nn.BCELoss()

        # epochs = 150 FIXME
    
    def train(self):
        # train
        self.loss_log2 = []
        self.nnet.train()
        tbar = tqdm(range(self.epochs))
        for epoch in tbar:
            for x, c, w, z in self.loader_train:
                # cuda
                if torch.cuda.is_available():
                    x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                # forward pass
                h = self.nnet(x)
                wp = torch.sigmoid(h)
                loss = self.bceloss(wp, w) # loss
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # log
                self.loss_log2.append(loss.item())
                tbar.set_description("Epoch: {:2}, Loss: {:3.4f}".format(epoch, loss.item()))
            # scheduled learning rate
            if (epoch == 90) or (epoch == 120):
                for g in self.optimizer.param_groups:
                    g['lr'] /= 10
        self.epoch = epoch
    
    def plot(self):
        # draw loss during training
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_log2, color="c")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlim(-300, len(self.loss_log2)+300)
        plt.xlabel("Iters", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Learning Curve on Training Set", fontsize=12)
        plt.show()
        print("Test set:")
        # init data
        data = {"Accuracy":[], "Optimal":[]}
        # eval
        self.nnet.eval()
        for x, c, w, z in tqdm(self.loader_test):
            # cuda
            if next(self.nnet.parameters()).is_cuda:
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # predict
            with torch.no_grad(): # no grad
                cp = self.nnet(x)
            # to numpy
            c = c.to("cpu").detach().numpy()
            w = w.to("cpu").detach().numpy()
            z = z.to("cpu").detach().numpy()
            cp = cp.to("cpu").detach().numpy()
            # solve
            for i in range(cp.shape[0]):
                # sol for pred cost
                self.optmodel.setObj(cp[i])
                wpi, _ = self.optmodel.solve()
                # obj with true cost
                zpi = np.dot(wpi, c[i])
                # round
                zpi = zpi.round(1)
                zi = z[i,0].round(1)
                # regret
                regret = (zpi - zi).round(1)
                # accuracy
                data["Accuracy"].append((abs(wpi - w[i]) < 0.5).mean())
                # optimal
                data["Optimal"].append(abs(regret) < 1e-2)
        # dataframe
        df2 = pd.DataFrame.from_dict(data)
        # print
        time.sleep(1)
        print("Path Accuracy: {:.2f}%".format(df2["Accuracy"].mean()*100))
        print("Optimality Ratio: {:.2f}%".format(df2["Optimal"].mean()*100))

        print("Train set:")
        # init data
        data = {"Accuracy":[], "Optimal":[]}
        # eval
        self.nnet.eval()
        for x, c, w, z in tqdm(self.loader_train):
            # cuda
            if next(self.nnet.parameters()).is_cuda:
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # predict
            with torch.no_grad(): # no grad
                cp = self.nnet(x)
            # to numpy
            c = c.to("cpu").detach().numpy()
            w = w.to("cpu").detach().numpy()
            z = z.to("cpu").detach().numpy()
            cp = cp.to("cpu").detach().numpy()
            # solve
            for i in range(cp.shape[0]):
                # sol for pred cost
                self.optmodel.setObj(cp[i])
                wpi, _ = self.optmodel.solve()
                # obj with true cost
                zpi = np.dot(wpi, c[i])
                # round
                zpi = zpi.round(1)
                zi = z[i,0].round(1)
                # regret
                regret = (zpi - zi).round(1)
                # accuracy
                data["Accuracy"].append((abs(wpi - w[i]) < 0.5).mean())
                # optimal
                data["Optimal"].append(abs(regret) < 1e-2)
        # dataframe
        df2 = pd.DataFrame.from_dict(data)
        # print
        time.sleep(1)
        print("Path Accuracy: {:.2f}%".format(df2["Accuracy"].mean()*100))
        print("Optimality Ratio: {:.2f}%".format(df2["Optimal"].mean()*100))