from torch.utils.data import Dataset
import torch


class mapDataset(Dataset):
    def __init__(self, tmaps, costs, paths):
        self.tmaps = tmaps
        self.costs = costs
        self.paths = paths
        self.objs = (costs * paths).sum(axis=(1, 2)).reshape(-1, 1)

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, ind):
        return (
            torch.FloatTensor(
                self.tmaps[ind].transpose(2, 0, 1) / 255
            ).detach(),  # image
            torch.FloatTensor(self.costs[ind]).reshape(-1),
            torch.FloatTensor(self.paths[ind]).reshape(-1),
            torch.FloatTensor(self.objs[ind]),
        )
