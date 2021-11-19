import torch


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, first, second=None, values=None):
        self.first = first
        self.second = second
        self.values = values

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.first.items()}
        if self.second is not None:
            for key, val in self.second.items():
                item['comparison_{}'.format(key)] = val[idx].clone().detach()
        if self.values is not None:
            item['values'] = self.values[idx]
        return item

    def __len__(self):
        return len(self.values)