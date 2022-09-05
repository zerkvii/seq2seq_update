import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim//n_heads
        self.q = nn.Linear(hid_dim, hid_dim)
        self.k = nn.Linear(hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, hid_dim)
        self.o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.LongTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)
        # K V is the same size while Q may not

        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        energy = (Q@K.permute(0, 1, 3, 2))/self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)

        x = self.dropout(attention) @ V
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.o(x)
        return x, attention


class PositionwiseForwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_att_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attn = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseForwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.self_att_layer_norm(src+self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src+self.dropout(_src))
        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=50):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(
            hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.FloatTensor([hid_dim]).to(device)

    def forward(self, src, src_mask):
        # batch first
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.dropout((self.embedding(src)*self.scale) +
                           self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device) -> None:
        super().__init__()
        self.multi_att = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.multi_att_norm = nn.LayerNorm(hid_dim)
        self.feed_forward_layer = PositionwiseForwardLayer(
            hid_dim, pf_dim, dropout)
        self.feed_norm = nn.LayerNorm(hid_dim)

        self.encoder_attn = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, en_src, trg_mask, enc_mask):

        _trg, _ = self.multi_att(trg, trg, trg, trg_mask)
        trg = self.multi_att_norm(trg+self.dropout(_trg))

        _trg, att = self.encoder_attn(trg, en_src, en_src, enc_mask)
        trg = self.encoder_norm(trg+self.dropout(_trg))

        _trg = self.feed_forward_layer(trg)
        trg = self.feed_norm(trg+self.dropout(_trg))

        return trg, att


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=50):
        super().__init__()
        self.device = device

        self.embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(
            hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, enc_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout(self.embedding(
            trg)*self.scale + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, enc_mask)
        output = self.fc_out(trg)
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(
            trg_len, trg_len)).bool().to(self.device)
        # batch_size 1 seq_len seq_len
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
