from pathlib import Path
import torch
import tensorflow as tf
from loguru import logger
from torch.utils.data import Dataset
from typing import List, Tuple
from scipy.io import arff
Tensor = torch.Tensor



def keep_subdirs_only(path: Path) -> None:
    files = [file for file in path.iterdir() if file.is_file()]
    for file in files:
        file.unlink()


def get_eeg(data_dir: Path) -> Path:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"  # noqa E501
    datapath = tf.keras.utils.get_file(
        "eeg", origin=url, untar=False, cache_dir=data_dir
    )
    logger.info(f"Data is downloaded to {datapath}.")
    return datapath


def get_eeg_data(cache_dir: str = ".") -> Tuple[List[Path], List[Path]]:
    datapath = Path(cache_dir)
    if datapath.exists():
        logger.info(f"{datapath} already exists, skipping download")
    else:
        logger.info(f"{datapath} not found on disk, downloading")
        get_eeg(cache_dir)
    
    data = arff.loadarff(cache_dir)
    return data


class eegDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = len(data)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # get a single item
        item = self.data[idx]
        x = [item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13]]
        if item[14] == b'0':
            y = 0
        else:
            y = 1

        return x, y


class eegDatasetWindowed(Dataset):
    def __init__(self, data, windowsize : int):
        self.data = data
        self.size = len(data)
        self.window = windowsize
    
    def __len__(self):
        return self.size - self.window

    def __getitem__(self, idx):
        # get a single item
        item = self.data[idx:idx + self.window]
        x = item[:-self.window,:]
        y = item[-self.window:,:].squeeze(-1) # squeeze will remove dimension -1 if possible.

        return x, y


class BaseDataset:
    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        random.shuffle(self.paths)
        self.dataset = []
        self.process_data()

    def process_data(self) -> None:
        # this needs to be implemented if you want to use the BaseDataset
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.dataset[idx]


class TSDataset(BaseDataset):
    # this is called inheritance.
    # we get all the methods from the BaseDataset for free
    # Only thing we need to do is implement the process_data method
    def process_data(self) -> None:
        for file in tqdm(self.paths):
            x_ = np.genfromtxt(file)[:, 3:]
            x = torch.tensor(x_).type(torch.float32)
            y = torch.tensor(int(file.parent.name) - 1)
            self.dataset.append((x, y))

