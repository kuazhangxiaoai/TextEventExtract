import os
import json

from torch.utils.data import DataLoader, Dataset

class DuEEDataset(Dataset):
    """
        This is dataset class for reading DuEE dataset
    """
    def __init__(self, cfg:dict):
        super().__init__()
        _, self.root_path, self.mode = cfg.values()
        self.train_file = os.path.join(self.root_path, 'train.json')
        self.dev_file = os.path.join(self.root_path, 'dev.json')
        with open(self.train_file) as f:
            lines = f.readlines()
            self.train_list = [json.loads(line) for line in lines]

        with open(self.dev_file) as f:
            lines = f.readlines()
            self.dev_list = [json.loads(line) for line in lines]

        assert self.mode in ["train", "val"]

    def load_one_item(self, index, mode):
        if mode == 'train':
            text_info, event_obj = self.train_list[index]['text'], self.train_list[index]['event_list']
        elif mode == 'val':
            text_info, event_obj = self.dev_list[index]['text'], self.dev_list[index]['event_list']
        return text_info, event_obj

    def __getitem__(self, index):
        return self.load_one_item(index, self.mode)

    def __len__(self):
        return len(self.train_list) if self.mode == 'train' else len(self.dev_list)

    @staticmethod
    def collate_fn(batch):
        new_batch = {
            "sentences": [],
            "augments": [],
        }
        for i, b in enumerate(batch):
            sentence, argument = b[0], b[1]
            new_batch['sentences'].append(sentence)
            new_batch['augments'].append(argument)
        return new_batch




