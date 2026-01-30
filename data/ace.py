import os
import json

from torch.utils.data import DataLoader, Dataset

class ACEDataset(Dataset):
    """
        This is dataset class for reading DuEE dataset
    """
    def __init__(self, root_path, mode):
        super().__init__()
        self.root_path = root_path
        self.mode = mode

        assert self.mode in ["train", "val", "test"]


