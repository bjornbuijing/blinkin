from __future__ import annotations
import random
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

Tensor = torch.Tensor


class BaseListDataset:
    """Base class for loading list data"""

    def __init__(self, data: list):
        self.data = data
        self.dataset = []
        self.process_data()

    def process_data(self) -> None:
        # abstract function which needs to be inherited
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.dataset[idx]


class BaseDataIterator:
    def __init__(self, dataset: BaseListDataset, batchsize: int):
        self.dataset = dataset
        self.batchsize = batchsize
        self.curindex = 0

    def __len__(self) -> int:
        # the lenght is the amount of batches
        return int(len(self.dataset) / self.batchsize)

    def __iter__(self) -> BaseDataIterator:
        # initialize index
        self.index = 0
        self.index_list = torch.randperm(len(self.dataset))
        return self

    def batchloop(self) -> Tuple[Tensor, Tensor]:
        X = []  # noqa N806
        Y = []  # noqa N806
        # fill the batch
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            X.append(x)
            Y.append(y)
            self.index += 1
        return X, Y

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.dataset) - self.batchsize):
            X, Y = self.batchloop()
            return X, Y
        else:
            raise StopIteration


class EEGBufferIterator:
    def __init__(self, dataset: BaseListDataset, batchsize: int, horizon: int):
        self.dataset = dataset
        self.batchsize = batchsize
        self.horizon = horizon
        self.curindex = 0
        self.index = 0
        self.sortedlist = []
        self.processdata()

    def processdata(self) -> None:
        #  Load everything into chunks
        self.sortedlist = []
        dumx, currenty = self.dataset[0]

        curchunk = []
        for x, y in self.dataset:
            if currenty == y:
                curchunk.append([x, y])
            else:
                self.sortedlist.append(curchunk)
                curchunk = []
                currenty = y

    def __len__(self) -> int:
        # the lenght is the amount of batches
        return int(len(self.dataset) / self.batchsize)

    def __iter__(self) -> BaseDataIterator:
        # initialize index
        return self

    def batchloop(self, id: int) -> Tuple[Tensor, Tensor]:
        randlist = self.sortedlist[id]
        print(len(randlist))
        batchlist = []
        for i in range(0, self.batchsize):
            horizonlist = []
            for j in range(0, self.horizon):
                if (i + j) < len(randlist):
                    x, y = randlist[i + j]
                    horizonlist.append(x)
                else:
                    break
            print(len(horizonlist))
            batchlist.append(pad_sequence(horizonlist, batch_first=True, padding_value=0)) # noqa E506          

        return pad_sequence(batchlist, batch_first=True, padding_value=0)

    def __next__(self) -> Tuple[Tensor, Tensor]:
        i = random.randint(0, len(self.sortedlist))
        return self.batchloop(i)
