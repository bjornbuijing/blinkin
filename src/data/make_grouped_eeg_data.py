from __future__ import annotations
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
import torch
Tensor = torch.Tensor


class BaseListDataset():
    """Base class for loading list data
    """
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


class SwitchIterator:
    def __init__(self, dataset: BaseListDataset):
        self.dataset = dataset
        self.curindex = 0
        self.index = 0

    def __len__(self) -> int:
        # the lenght is the amount of batches
        return int(len(self.dataset) / self.batchsize)

    def __iter__(self) -> BaseDataIterator:
        # initialize index

        return self

    def batchloop(self) -> Tuple[Tensor, Tensor]:
        X = []  # noqa N806
        Y = []  # noqa N806
        # fill the batch
        x, y = self.dataset[int(self.index)]
        X.append(x)
        Y.append(y)
        self.index = self.index + 1
        currentY = y
        while y == currentY:
            if self.index + 1 < len(self.dataset):
                X.append(x)
                Y.append(y)
                self.index = self.index + 1
                x, y = self.dataset[int(self.index)]
            else:
                self.index = 0
                break

        return X, Y

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index < (len(self.dataset)):
            X, Y = self.batchloop()
            return X, Y
        else:
            raise StopIteration


class SwitchPaddedDatagenerator(SwitchIterator):
    # again, we inherit everything from the baseclass
    def __init__(self, dataset: BaseListDataset) -> None:
        # we initialize the super class BaseDataIterator
        # we now have everything the BaseDataIterator can do, for free
        super().__init__(dataset)

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index < (len(self.dataset)):
            X, Y = self.batchloop()
            # we just want to add padding
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_, torch.tensor(Y)
        else:
            raise StopIteration
