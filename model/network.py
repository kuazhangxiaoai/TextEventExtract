import torch
import torch.nn as nn
from transformers import AutoModel
from utils.utils import split_tensor_sliding_window

class TriNet(nn.Module):
    def __init__(self, bert_name, bert_hid_size, mode, tri_type_num, split, device):
        super().__init__()
        self.berter = AutoModel.from_pretrained(bert_name, cache_dir="./cache/", output_hidden_states=True)
        self.training = mode == 'train'
        self.split = split
        self.device = device
        self.tri_type_num = tri_type_num
        self.eve_embedding = nn.Embedding(self.tri_type_num + 1, bert_hid_size, padding_idx=self.tri_type_num).to(device)
        self.berter.to(device) # cuda or cpu

    def bert(self, x, attn_mask):
        if self.split: #如果长度长度超过512 则进行切割。
            x_splits, attn_mask_splits = split_tensor_sliding_window(x, attn_mask)
            bert_embedings = []
            for x_seg, atten_seg in zip(x_splits, attn_mask_splits):
                bert_embedings.append(self.berter(input_ids=x_seg, attention_mask=atten_seg)[0])
            return torch.cat(bert_embedings, dim=1)
        else:
            return self.berter(x, attn_mask)

    def forward(self, x, attn_mask, word_mask1d, word_mask2d, pos_events, neg_events):
        bert_embeding = self.bert(x, attn_mask)
        L = word_mask1d.size(1)
        B, _, H = bert_embeding.size()
        bert_embeding = bert_embeding[:, 1:1 + L]
        global_tri_embedings = self.eve_embedding(torch.arange(self.tri_type_num).long().cuda()) #all event embeding
        pos_events[pos_events<0] = self.tri_type_num

        if self.training:
            eve_emb = self.eve_embedding(bert_embeding)

        return x