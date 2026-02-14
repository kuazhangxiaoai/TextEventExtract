import torch

from model.network import TriNet
from utils import load_config, make_config
from data import dataset_map, build_dataloader


class Trainer:
    "This is trainer"
    def __init__(self, cfg):
        self.cfg = cfg

    def build_dataset(self):
        dataset_cfg = self.cfg['data']['dataset']
        dataloader_cfg = self.cfg['data']['dataloader']
        name, root_path = dataset_cfg['name'], dataset_cfg['root_path']
        self.dataset = dataset_map[name](dataset_cfg)
        self.dataloader = build_dataloader(self.dataset,
                                           dataloader_cfg['batch_size'],
                                           dataloader_cfg['workers'],
                                           dataloader_cfg['shuffle'])
        self.tri_type_num = self.dataset.event_types_num



    def setup_train(self):
        hyp = self.cfg['hyp']['train']
        self.deivce = torch.device(hyp['device'])
        self.epoach = hyp['epoach']
        self.build_dataset()
        self.setup_model()

    def setup_model(self):
        bert_name = self.cfg['data']['dataset']['bert']
        bert_hid_size = self.cfg['model']['bert_hid_size']
        self.model = TriNet(bert_name, bert_hid_size, 'train', self.tri_type_num, self.cfg['model']['split'], self.deivce)


    def train(self):
        self.setup_train()
        for i in range(self.epoach):
            for batch_i, batch in enumerate(self.dataloader):
                inputs, atten_mask, word_mask1d, word_mask2d, tri_targets, pos_events, neg_events = batch
                pred = self.model(inputs.to(self.deivce),
                                  atten_mask.to(self.deivce),
                                  word_mask1d.to(self.deivce),
                                  word_mask2d.to(self.deivce),
                                  pos_events.to(self.deivce),
                                  neg_events.to(self.deivce))
                #print(batch)


if __name__ == '__main__':
    cfg = make_config('config/data/ace.yml',
                      'config/model/trinet.yml',
                      'config/config.yml')
    trainer = Trainer(cfg)
    trainer.train()
    print('done')