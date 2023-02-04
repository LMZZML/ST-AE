import torch.nn as nn
from util import *
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = attention_dropout
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = F.dropout(attention, self.dropout, self.training)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention_T(nn.Module):

    def __init__(self, model_dim=2, num_heads=8, dropout=0.0):
        super(MultiHeadAttention_T, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.linear_k = nn.Linear(model_dim, num_heads * model_dim)
        self.linear_v = nn.Linear(model_dim, num_heads * model_dim)
        self.linear_q = nn.Linear(model_dim, num_heads * model_dim)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(num_heads * model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input, attn_mask=None):
        residual = input

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        model_dim = self.model_dim
        batch_size = input.size(0)

        # linear projection
        key = self.linear_k(input)
        value = self.linear_v(input)
        query = self.linear_q(input)

        # split by heads
        key = key.view(batch_size * num_heads, -1, model_dim)
        value = value.view(batch_size * num_heads, -1, model_dim)
        query = query.view(batch_size * num_heads, -1, model_dim)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        # scale = (key.size(-1) // num_heads) ** -0.5
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, 1, -1, model_dim * num_heads)
        # final linear projection
        output = self.linear_final(context)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class gated_selfatt_T(nn.Module):

    def __init__(self, num_node=170, seq_len=2, num_heads=6, dropout=0.5):
        super(gated_selfatt_T, self).__init__()
        self.encoder_s = MultiHeadAttention_T(model_dim=seq_len, num_heads=num_heads, dropout=dropout)
        self.encoder_t = MultiHeadAttention_T(model_dim=num_node, num_heads=num_heads, dropout=dropout)
        self.W_s = nn.Linear(seq_len, seq_len)
        self.W_t = nn.Linear(seq_len, seq_len)

    def forward(self, input):
        hidden_state_s = self.encoder_s(input)
        hidden_state_t = self.encoder_t(input.transpose(2,3)).transpose(2,3)
        z = torch.sigmoid(self.W_s(hidden_state_s)+self.W_t(hidden_state_t))
        hidden_state = torch.mul(z, hidden_state_s)+torch.mul(1-z, hidden_state_t)
        output = hidden_state + input
        return output

