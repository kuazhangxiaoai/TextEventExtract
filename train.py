from utils import load_config
from data import dataset_map, build_dataloader

class Trainer:
    "This is trainer"
    def __init__(self, cfg):
        self.cfg = cfg

    def build_dataset(self):
        data_cfg = self.cfg['data']
        dataloader_cfg = self.cfg['dataloader']
        name, root_path = data_cfg['name'], data_cfg['root_path']
        self.dataset = dataset_map[name](data_cfg)
        self.dataloader = build_dataloader(self.dataset,
                                           dataloader_cfg['batch_size'],
                                           dataloader_cfg['workers'],
                                           dataloader_cfg['shuffle'])

    def setup_train(self):
        hyp = self.cfg['hyp']
        self.epoach = hyp['epoach']
        self.build_dataset()

    def train(self):
        self.setup_train()
        for i in range(self.epoach):
            for batch_i, batch in enumerate(self.dataloader):
                print(batch)






if __name__ == '__main__':
    cfg = load_config('config/ace.yml')
    trainer = Trainer(cfg)
    trainer.train()
    print('done')