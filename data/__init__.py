from data.ace import ACEDataset
from data.duee import DuEEDataset
from data.utils import build_dataloader, get_files_from_dir

dataset_map = {
    "DuEE": DuEEDataset,
    "ACE": ACEDataset
}

__all__ = [
    "ACEDataset",
    "DuEEDataset"
]

