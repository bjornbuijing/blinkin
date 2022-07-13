from __future__ import annotations

from typing import Tuple, List, Any

import torch
from torch.utils.data import Dataset

Tensor = torch.Tensor


class BaseListSlicedDataset(Dataset):
    """Base class for loading list data"""

    def __init__(self: BaseListSlicedDataset, data: list, horizon: int) -> None: # noqa E501
        self.data = data
        self.dataset = torch.tensor
        self.horizon = horizon
        self.process_data()
        self.size = len(self.dataset)

    def process_data(self: BaseListSlicedDataset) -> None:
        newlist = []
        for record in self.data:
            newlist.append(
                [
                    record[0],
                    record[1],
                    record[2],
                    record[3],
                    record[4],
                    record[5],
                    record[6],
                    record[7],
                    record[8],
                    record[9],
                    record[10],
                    record[11],
                    record[12],
                    record[13],
                    float(record[14]),
                ]
            )

        self.dataset = torch.FloatTensor(newlist)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # get a single item
        # item = self.dataset[idx]
        # slice off the horizon
        # x = self.dataset[:-self.horizon, :]
        # squeeze will remove dimension -1 if possible.
        # y = self.dataset[-self.horizon:, :].squeeze(-1)
        i = 0
        x = []  # type: List[Any]
        y = []  # type: List[Any]
        while i < self.horizon and idx + i < self.size:
            x.append(
                [
                    self.data[idx + i][0],
                    self.data[idx + i][1],
                    self.data[idx + i][2],
                    self.data[idx + i][3],
                    self.data[idx + i][4],
                    self.data[idx + i][5],
                    self.data[idx + i][6],
                    self.data[idx + i][7],
                    self.data[idx + i][8],
                    self.data[idx + i][9],
                    self.data[idx + i][10],
                    self.data[idx + i][11],
                    self.data[idx + i][12],
                    self.data[idx + i][13],
                ]
            )
            y.append([int(self.data[idx + i][14])])
            i = i + 1
        x = torch.FloatTensor(x)
        return x, y
