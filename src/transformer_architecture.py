import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer_components import GRN

## Multi Head Sparse Attention

class SparseMultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, top_k, dropout = 0.1):
		super().__init__()
		assert d_model % n_heads == 0
		self.d_model = d_model
		self.n_heads = n_heads
		self.d_k = d_model // n_heads
		self.top_k = top_k

		self.W_q = nn.Linear(d_model, d_model)
		self.W_k = nn.Linear(d_model, d_model)
		self.W_v = nn.Linear(d_model, d_model)
		self.W_o = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		n_firms = x.shape[0]
		x = x.unsqueeze(0)

		Q = self.W_q(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		K = self.W_k(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		V = self.W_v(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)

		scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

		k = min(self.top_k, n_firms)
		topk_vals, _ = scores.topk(k, dim = -1)
		threshold = topk_vals[..., -1:].detach()
		mask = scores < threshold
		scores = scores.masked_fill(mask, float("-inf"))

		attn_weights = F.softmax(scores, dim = -1)
		attn_weights = self.dropout(attn_weights)

		context = torch.matmul(attn_weights, V)
		context = context.transpose(1, 2).contiguous().view(1, n_firms, self.d_model)
		out = self.W_o(context).squeeze(0)

		return out, attn_weights.squeeze(0)
	
## Transformer Encoder Block

class TransformerBlock(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, top_k, dropout = 0.1):
		super().__init__()
		self.norm1 = nn.LayerNorm(d_model)
		self.attention = SparseMultiHeadAttention(d_model, n_heads, top_k, dropout)
		self.grn = GRN(d_model, d_ff, dropout)

	def forward(self, x):
		normed = self.norm1(x)
		attn_out, attn_weights = self.attention(normed)
		x = x + attn_out
		x = self.grn(x)
		return x, attn_weights