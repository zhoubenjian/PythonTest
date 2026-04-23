import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# 1. 位置编码
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# --------------------------
# 2. 缩放点积注意力
# --------------------------
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# --------------------------
# 3. 多头注意力
# --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.w_o(attn_out)
        return out


# --------------------------
# 4. 前馈网络 FFN
# --------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# --------------------------
# 5. 单个 Encoder Layer
# --------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# --------------------------
# 6. 完整 Encoder
# --------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x


# --------------------------
# 测试
# --------------------------
if __name__ == "__main__":
    vocab_size = 1000
    d_model = 128
    n_layers = 6
    n_heads = 8
    d_ff = 512

    encoder = TransformerEncoder(vocab_size, d_model, n_layers, n_heads, d_ff)
    x = torch.randint(0, vocab_size, (2, 10))
    out = encoder(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", out.shape)