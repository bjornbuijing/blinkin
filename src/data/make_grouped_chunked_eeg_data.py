from __future__ import annotations

from typing import Tuple, List, Any

import torch
from torch.nn.utils.rnn import pad_sequence

Tensor = torch.Tensor


class BaseListDataset:
    """Base class for loading list data"""

    def __init__(self, data: list) -> None:
        self.data = data
        self.dataset = []  # type: List[Tuple]
        self.process_data()

    def process_data(self) -> None:
        # abstract function which needs to be inherited
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return self.dataset[idx]


class BaseDataIterator:
    def __init__(
        self, dataset: BaseListDataset, batchsize: int
    ) -> None:  # noqa E501
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

    def batchloop(self) -> Tuple[List[Tensor], List[int]]:
        X = []  # noqa N806
        Y = []  # noqa N806
        # fill the batch
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            X.append(x)
            Y.append(y)
            self.index += 1
        return X, Y

    def __next__(self) -> Tuple[List[Tensor], List[int]]:
        if self.index <= (len(self.dataset) - self.batchsize):
            X, Y = self.batchloop()
            return X, Y
        else:
            raise StopIteration


class EEGChunkIterator:
    def __init__(
        self, dataset: BaseListDataset, batchsize: int, horizon: int
    ) -> None:  # noqa E501
        self.dataset = dataset
        self.batchsize = batchsize
        self.horizon = horizon
        self.curindex = 0
        self.index = 0
        #  Load everything into chunks

    def __len__(self) -> int:
        # the lenght is the amount of batches
        return int(len(self.dataset) / self.batchsize)

    def __iter__(self) -> EEGChunkIterator:
        # initialize index
        return self

    def batchloop(self) -> List[Tensor]:
        X = []  # noqa N806
        Y = []  # noqa N806
        # fill the batch
        x, y = self.dataset[int(self.index)]
        count = 1
        batchlist = []
        self.index = self.index + 1
        currenty = y
        while y == currenty and count <= self.batchsize:
            if self.index == self.__len__:
                break
            else:
                # We need too chunk here
                horizoncount = 0
                horizonlist = []
                while horizoncount < self.horizon:
                    x, y = self.dataset[int(self.index) + horizoncount]
                    if y == currenty:
                        horizonlist.append(x)  # noqa E506
                        horizoncount = horizoncount + 1
                    else:
                        break
            batchlist.append(
                pad_sequence(horizonlist, batch_first=True, padding_value=0)
            )  # noqa E506
            count = count + 1
            self.index = self.index + 1
            x, y = self.dataset[int(self.index)]

        # we need to add padding
        # return X, Y
        return batchlist

    def __next__(self) -> Tensor:
        if self.index <= (len(self.dataset)):
            X = self.batchloop()  # noqa N806
            # we just want to add padding
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_
        else:
            raise StopIteration
