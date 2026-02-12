import math
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class Gate(nn.Module):
    def __init__(self, hid_size, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(hid_size * 2, hid_size)
        self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(hid_size, hid_size)

    def forward(self, x, y):
        '''
        :param x: B, L, K, H
        :param y: B, L, K, H
        :return:
        '''
        o = torch.cat([x, y], dim=-1)
        o = self.dropout(o)
        gate = self.linear(o)
        gate = torch.sigmoid(gate)
        o = gate * x + (1 - gate) * y
        # o = F.gelu(self.linear2(self.dropout(o)))
        return o

class AdaptiveFusion(nn.Module):
    def __init__(self, hid_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.q_linear = nn.Linear(hid_size, hid_size)
        self.k_linear = nn.Linear(hid_size, hid_size * 2)

        self.factor = math.sqrt(hid_size)

        self.gate1 = Gate(hid_size, dropout=dropout)
        self.gate2 = Gate(hid_size, dropout=dropout)

    def forward(self, x, s, g):
        # x [B, L, H]
        # s [B, K, H]
        # g [B, N, H]
        # x = self.dropout(x)
        # s = self.dropout(s)
        q = self.q_linear(x)
        k_v = self.k_linear(g)
        k, v = torch.chunk(k_v, chunks=2, dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.factor
        # scores = self.dropout(scores)
        scores = torch.softmax(scores, dim=-1)
        g = torch.bmm(scores, v)
        g = g.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        h = x.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        s = s.unsqueeze(1).expand(-1, x.size(1), -1, -1)

        h = self.gate1(h, g)
        h = self.gate2(h, s)
        return h

class FFNN(nn.Module):
    def __init__(self, input_dim, hid_dim, cls_num, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, cls_num)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x

class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s

