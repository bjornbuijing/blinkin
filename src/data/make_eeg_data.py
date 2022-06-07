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

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.dataset[idx]


class EEGListDataset(BaseListDataset):
    """Processes data for EEG Lists

    Args:
        BaseListDataset (_type_): base class for list data
    """
    def process_data(self) -> None:
        for record in self.data:
            x = torch.tensor([record[0], record[1], record[2], record[3],
                             record[4], record[5], record[6], record[7],
                             record[8], record[9], record[10], record[11],
                             record[12], record[13]], dtype=float)
            y = torch.tensor(int(record[14]))
            self.dataset.append((x, y))


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


class PaddedDatagenerator(BaseDataIterator):
    # again, we inherit everything from the baseclass
    def __init__(self, dataset: BaseListDataset, batchsize: int) -> None:
        # we initialize the super class BaseDataIterator
        # we now have everything the BaseDataIterator can do, for free
        super().__init__(dataset, batchsize)

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.dataset) - self.batchsize):
            X, Y = self.batchloop()
            # I do not have a clue why
            # this function returns a tensor
            # and torch.tensor(X) gives me an error
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_, torch.tensor(Y)
        else:
            raise StopIteration
