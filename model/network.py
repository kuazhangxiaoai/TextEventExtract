import torch
import torch.nn as nn
from transformers import AutoModel
from utils.utils import split_tensor_sliding_window
from model.block import AdaptiveFusion
from torch.nn.utils.rnn import pad_sequence

class OneEventExtracter(nn.Module):
    """
        The oneEE model implement
    """
    def __init__(self, config):
        super(OneEventExtracter, self).__init__()
        self.inner_dim = config.tri_hid_size
        self.tri_hid_size = config.tri_hid_size
        self.eve_hid_size = config.eve_hid_size
        self.event_num = config.tri_label_num
        self.role_num = config.rol_label_num
        self.teacher_forcing = True
        self.gamma = config.gamma
        self.arg_hid_size = config.arg_hid_size
        # self.layers = config.layers
        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)

        # self.tri_hid_size = 256
        # self.arg_hid_size = 384

        self.dropout = nn.Dropout(config.dropout)
        self.tri_linear = nn.Linear(config.bert_hid_size, self.tri_hid_size * 2)
        self.arg_linear = nn.Linear(config.bert_hid_size, self.arg_hid_size * 2)
        self.role_linear = nn.Linear(config.bert_hid_size, config.eve_hid_size * config.rol_label_num * 2)
        self.eve_embedding = nn.Embedding(self.event_num + 1, config.bert_hid_size, padding_idx=self.event_num) #.from_pretrained(self.reset_event_parameters(config.vocab, config.tokenizer), freeze=False)
        # self.layer_norm = LayerNorm(config.bert_hid_size, config.bert_hid_size, conditional=True)

        self.gate = AdaptiveFusion(config.bert_hid_size, dropout=config.dropout)

    def reset_event_parameters(self, vocab, tokenizer):
        labels = [vocab.tri_id2label[i] for i in range(self.event_num)]
        inputs = tokenizer(labels)
        input_ids = pad_sequence([torch.LongTensor(x) for x in inputs["input_ids"]], True)
        attention_mask = pad_sequence([torch.BoolTensor(x) for x in inputs["attention_mask"]], True)
        mask = pad_sequence([torch.BoolTensor(x[1:-1]) for x in inputs["attention_mask"]], True)
        bert_embs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embs = bert_embs[0][:, 1:-1]
        min_value = bert_embs.min().item()
        bert_embs = torch.masked_fill(bert_embs, mask.eq(0).unsqueeze(-1), min_value)
        bert_embs, _ = torch.max(bert_embs, dim=1)
        bert_embs = torch.cat([bert_embs, torch.zeros((1, bert_embs.size(-1)))], dim=0)
        return bert_embs.detach()


    def _sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.cuda()
        return embeddings

    def _pointer(self, qw, kw, word_mask2d):
        B, L, K, H = qw.size()
        pos_emb = self._sinusoidal_position_embedding(B, L, H)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        grid_mask2d = word_mask2d.unsqueeze(1).expand(B, K, L, L).float()
        logits = logits * grid_mask2d - (1 - grid_mask2d) * 1e12
        return logits

    def forward(self, inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, event_idx=0):
        """
        :param inputs: [B, L]
        :param att_mask: [B, L]
        :param word_mask1d: [B, L]
        :param word_mask2d: [B, L, L]
        :param span_labels: [B, L, L, 2], [..., 0] is trigger span label, [..., 1] is argument span label
        :param tri_labels: [B, L, L, C]
        :param event_mask: [B, L]
        :param prob: float (0 - 1)
        :return:
        """
        outputs = {}


        L = word_mask1d.size(1)

        bert_embs = self.bert(input_ids=inputs, attention_mask=att_mask)  #
        bert_embs = bert_embs[0]

        B, _, H = bert_embs.size()

        bert_embs = bert_embs[:, 1:1 + L]

        B, L, H = bert_embs.size()

        eve_embs = self.eve_embedding(torch.arange(self.event_num).long().cuda())
        eve_embs = eve_embs[None, ...].expand(B, -1, -1)

        if self.training:
            # x = bert_embs
            # y = self.eve_embedding(torch.LongTensor([event_idx]).cuda())[None, ...].expand(B, L, -1)
            # cond_bert_embs = self.gate(x, y)
            #
            # drop_bert_embs = self.dropout(cond_bert_embs)
            #
            # span_reps = self.span_linear(drop_bert_embs).view(B, L, 1, self.inner_dim * 4)
            #
            # tri_qw, tri_kw, arg_qw, arg_kw = torch.chunk(span_reps, 4, dim=-1)
            #
            # tri_logits = self._pointer(tri_qw, tri_kw, word_mask2d).permute(0, 2, 3, 1).squeeze()
            # arg_logits = self._pointer(arg_qw, arg_kw, word_mask2d).permute(0, 2, 3, 1).squeeze()

            x = bert_embs
            y = self.eve_embedding(event_idx)
            # y = torch.cat([y, g_eve_embs], dim=2)
            cond_bert_embs = self.gate(x, y, eve_embs)
            # cond_arg_embs = self.arg_gate(x, y)
            # cond_role_embs = self.role_gate(x, y)
            # drop_tri_embs = cond_bert_embs
            drop_tri_embs = self.dropout(cond_bert_embs)
            # drop_arg_embs = self.dropout(cond_bert_embs)
            # drop_role_embs = self.dropout(cond_bert_embs)

            tri_reps = self.tri_linear(drop_tri_embs).view(B, L, -1, self.tri_hid_size * 2)
            tri_qw, tri_kw = torch.chunk(tri_reps, 2, dim=-1)
            arg_reps = self.arg_linear(drop_tri_embs).view(B, L, -1, self.arg_hid_size * 2)
            arg_qw, arg_kw = torch.chunk(arg_reps, 2, dim=-1)

            tri_logits = self._pointer(tri_qw, tri_kw, word_mask2d).permute(0, 2, 3, 1)
            arg_logits = self._pointer(arg_qw, arg_kw, word_mask2d).permute(0, 2, 3, 1)

            role_reps = self.role_linear(drop_tri_embs).view(B, L, -1, self.eve_hid_size * 2)
            role_qw, role_kw = torch.chunk(role_reps, 2, dim=-1)

            # role_qw = self.role_linear1(drop_tri_embs).view(B, L, -1, self.eve_hid_size)
            # role_kw = self.role_linear2(drop_arg_embs).view(B, L, -1, self.eve_hid_size)
            role_logits = self._pointer(role_qw, role_kw, triu_mask2d).permute(0, 2, 3, 1).view(B, L, L, -1,
                                                                                                self.role_num)

            return tri_logits, arg_logits, role_logits
        else:
            x = bert_embs
            y = self.eve_embedding(torch.LongTensor([i for i in range(self.event_num)]).cuda()).unsqueeze(0).expand(B, -1, -1)
            # y = torch.cat([y, g_eve_embs], dim=2)
            cond_bert_embs = self.gate(x, y, eve_embs)
            # cond_arg_embs = self.arg_gate(x, y)
            # cond_role_embs = self.role_gate(x, y)
            # cond_tri_embs = torch.cat([cond_tri_embs, bert_embs.unsqueeze(2)], dim=2)
            # cond_arg_embs = torch.cat([cond_arg_embs, bert_embs.unsqueeze(2)], dim=2)
            # cond_bert_embs = torch.cat([cond_bert_embs, bert_embs.unsqueeze(2)], dim=2)

            # drop_tri_embs = self.dropout(cond_tri_embs)

            # span_reps = self.tri_linear(drop_tri_embs).view(B, L, -1, self.inner_dim * 4)
            #
            # tri_qw, tri_kw, arg_qw, arg_kw = torch.chunk(span_reps, 4, dim=-1)
            # drop_tri_embs = cond_bert_embs
            drop_tri_embs = self.dropout(cond_bert_embs)
            # drop_arg_embs = self.dropout(cond_bert_embs)
            # drop_role_embs = self.dropout(cond_bert_embs)

            tri_reps = self.tri_linear(drop_tri_embs).view(B, L, -1, self.tri_hid_size * 2)
            tri_qw, tri_kw = torch.chunk(tri_reps, 2, dim=-1)
            arg_reps = self.arg_linear(drop_tri_embs).view(B, L, -1, self.arg_hid_size * 2)
            arg_qw, arg_kw = torch.chunk(arg_reps, 2, dim=-1)

            tri_logits = self._pointer(tri_qw, tri_kw, word_mask2d).permute(0, 2, 3, 1)
            arg_logits = self._pointer(arg_qw, arg_kw, word_mask2d).permute(0, 2, 3, 1)

            role_reps = self.role_linear(drop_tri_embs).view(B, L, -1, self.eve_hid_size * 2)
            role_qw, role_kw = torch.chunk(role_reps, 2, dim=-1)

            # role_qw = self.role_linear1(drop_tri_embs).view(B, L, -1, self.eve_hid_size)
            # role_kw = self.role_linear2(drop_arg_embs).view(B, L, -1, self.eve_hid_size)
            role_logits = self._pointer(role_qw, role_kw, triu_mask2d).permute(0, 2, 3, 1).view(B, L, L, -1,
                                                                                                self.role_num)

            # tri_g_logits = tri_logits[..., -1]
            # arg_g_logits = arg_logits[..., -1]
            #
            # tri_logits = tri_logits[..., :-1]
            # arg_logits = arg_logits[..., :-1]

            # tri_g_b_index, tri_g_x_index, tri_g_y_index = ((tri_g_logits > 0).long() + word_mask2d.long()).eq(2).nonzero(as_tuple=True)
            # arg_g_b_index, arg_g_x_index, arg_g_y_index = ((arg_g_logits > 0).long() + word_mask2d.long()).eq(
            #     2).nonzero(as_tuple=True)

            tri_b_index, tri_x_index, tri_y_index, tri_e_index = ((tri_logits > 0).long() + word_mask2d[..., None].long()).eq(2).nonzero(as_tuple=True)  # trigger index

            arg_b_index, arg_x_index, arg_y_index, arg_e_index = ((arg_logits > 0).long() + word_mask2d[..., None].long()).eq(2).nonzero(as_tuple=True)  # trigger index

            role_b_index, role_x_index, role_y_index, role_e_index, role_r_index = (role_logits > 0).nonzero(
                as_tuple=True)

            # tri_g_b_index = torch.cat([tri_g_b_index, tri_b_index], dim=0)
            # tri_g_x_index = torch.cat([tri_g_x_index, tri_x_index], dim=0)
            # tri_g_y_index = torch.cat([tri_g_y_index, tri_y_index], dim=0)

            outputs["ti"] = torch.cat([x.unsqueeze(-1) for x in [ tri_b_index, tri_x_index, tri_y_index]],
                                      dim=-1).cpu().numpy()

            outputs["tc"] = torch.cat([x.unsqueeze(-1) for x in [tri_b_index, tri_x_index, tri_y_index, tri_e_index]],
                                      dim=-1).cpu().numpy()

            outputs["ai"] = torch.cat([x.unsqueeze(-1) for x in [arg_b_index, arg_x_index, arg_y_index, arg_e_index]],
                                      dim=-1).cpu().numpy()

            # outputs["ac"] = None
            outputs["ac"] = torch.cat([x.unsqueeze(-1) for x in [role_b_index, role_y_index, role_e_index, role_r_index]],
                                      dim=-1).cpu().numpy()

            # outputs["as"] = torch.cat([x.unsqueeze(-1) for x in [arg_g_b_index, arg_g_x_index, arg_g_y_index]],
            #                           dim=-1).cpu().numpy()



            return outputs

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
            events = torch.cat([pos_events, neg_events], dim=1)
            eve_emb = self.eve_embedding(events)
        else:
            events = torch.LongTensor([x for x in range(self.tri_type_num)]).to(self.device)

        return x