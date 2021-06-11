import torch
import math

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


class RNN(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=1024,
                 n_layers=1,
                 batch_size=1,
                 cls_num=1,
                 drop_prob=0.5,
                 device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.rnn_layer = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)

        self.linear = nn.Linear(hidden_dim, cls_num)
        self.dropout = nn.Dropout(drop_prob)
        self.sigmoid = nn.Sigmoid()
        self.n_layers, self.batch_size, self.hidden_dim, self.device = n_layers, batch_size, hidden_dim, device

    def forward(self, input_seq, hidden):
        valid_shape = input_seq.shape[0]
        if valid_shape != hidden.shape[1]:
            _input_seq = torch.zeros(self.batch_size, input_seq.shape[1], input_seq.shape[2])
            _input_seq[:input_seq.shape[0]] = input_seq
            input_seq = _input_seq.to(input_seq.device)
        out, hidden = self.rnn_layer(input_seq, hidden)
        predictions = self.linear(self.dropout(out))
        if valid_shape != hidden.shape[1]:
            predictions = predictions[:valid_shape]
        return self.sigmoid(predictions)[:, -1, 0], hidden

    def init_hidden(self):
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        return hidden_state


class LSTM(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=1024,
                 n_layers=1,
                 batch_size=1,
                 cls_num=1,
                 drop_prob=0.5,
                 device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        self.linear = nn.Linear(hidden_dim, cls_num)
        self.dropout = nn.Dropout(drop_prob)
        self.sigmoid = nn.Sigmoid()
        self.n_layers, self.batch_size, self.hidden_dim, self.device = n_layers, batch_size, hidden_dim, device

    def forward(self, input_seq, hidden):
        valid_shape = input_seq.shape[0]
        if valid_shape != hidden[1].shape[1]:
            _input_seq = torch.zeros(self.batch_size, input_seq.shape[1], input_seq.shape[2])
            _input_seq[:input_seq.shape[0]] = input_seq
            input_seq = _input_seq.to(input_seq.device)
        lstm_out, hidden = self.lstm_layer(input_seq, hidden)
        predictions = self.linear(self.dropout(lstm_out))
        if valid_shape != hidden[1].shape[1]:
            predictions = predictions[:valid_shape]
        return self.sigmoid(predictions)[:, -1, 0], hidden

    def init_hidden(self):
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        cell_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        return hidden_state, cell_state



class GRU(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=1024,
                 n_layers=1,
                 batch_size=1,
                 cls_num=1,
                 drop_prob=0.5,
                 device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.gru_layer = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)

        self.linear = nn.Linear(hidden_dim, cls_num)
        self.dropout = nn.Dropout(drop_prob)
        self.sigmoid = nn.Sigmoid()
        self.n_layers, self.batch_size, self.hidden_dim, self.device = n_layers, batch_size, hidden_dim, device

    def forward(self, input_seq, hidden):
        valid_shape = input_seq.shape[0]
        if valid_shape != self.batch_size:
            _input_seq = torch.zeros(self.batch_size, input_seq.shape[1], input_seq.shape[2])
            _input_seq[:input_seq.shape[0]] = input_seq
            input_seq = _input_seq.to(input_seq.device)
        lstm_out, hidden = self.gru_layer(input_seq, hidden)
        predictions = self.linear(self.dropout(lstm_out))
        if valid_shape != self.batch_size:
            predictions = predictions[:valid_shape]
        return self.sigmoid(predictions)[:, -1, 0], hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self,
                 input_dim=768,
                 n_layers=6,
                 heads=8,
                 dim_head=64,
                 hidden_dim=1024,
                 drop_prob=0,
                 cls_num=1,
                 *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(input_dim, Attention(input_dim, heads = heads, dim_head = dim_head, dropout = drop_prob))),
                Residual(PreNorm(input_dim, FeedForward(input_dim, hidden_dim, dropout = drop_prob)))
            ]))
        self.pos_embedding = PositionalEncoding(input_dim)
        self.sigmoid = nn.Sigmoid()
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cls_num)
        )

    def forward(self, x, mask = None):
        x = self.pos_embedding(x)
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        x = self.to_latent(x[:, 0])
        x = self.mlp_head(x)
        return self.sigmoid(x)[:, 0]

    def init_hidden(self):
        return None


def create_model(model='lstm'):
    if 'lstm' in model:
        return LSTM
    elif 'gru' in model:
        return GRU
    elif 'transformer' in model:
        return Transformer
    elif 'rnn' in model:
        return RNN

