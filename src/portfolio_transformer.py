import torch
import torch.nn as nn

from transformer_components import Time2Vec, build_encoder
from transformer_architecture import TransformerBlock
from configuration import *

class PortfolioTransformer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		n_k0 = len(K0_CHARS)
		n_k1 = len(K1_CHARS)
		n_miss = len(miss_flags)

		self.k0_encoder = build_encoder(
			config.encoding_variant, n_k0, config.d_model,
			ple_bins = config.ple_num_bins,
			periodic_freq = config.periodic_num_freq
		)
		self.k1_encoder = build_encoder(
			config.encoding_variant, n_k1, config.d_model,
			ple_bins = config.ple_num_bins,
			periodic_freq = config.periodic_num_freq
		)

		self.time2vec = Time2Vec(config.d_model)
		self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)
		self.miss_proj = nn.Linear(n_miss, config.d_model)

		self.blocks = nn.ModuleList([
			TransformerBlock(
				config.d_model, config.n_heads, config.d_ff,
				config.top_k_attention, config.dropout
			)
			for _ in range(config.n_layers)
		])

		self.head_3m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.n_classes))
		self.head_6m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.n_classes))
		self.head_12m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.n_classes))

		self.register_buffer("lag_positions", torch.tensor(LAG_POSITIONS, dtype = torch.float32))

	def _encode_firm_token(self, k0, k1, miss):
		n_firms = k0.shape[0]

		k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
		k0_token = k0_encoded.sum(dim = 1)

		k1_flat = k1.permute(0, 2, 1).reshape(n_firms * 6, -1)
		k1_encoded = self.k1_encoder(k1_flat)
		k1_encoded = k1_encoded.view(n_firms, 6, len(K1_CHARS), self.config.d_model)

		t2v_all = self.time2vec(self.lag_positions).unsqueeze(0).unsqueeze(2)
		k1_encoded = k1_encoded + t2v_all

		k1_token = k1_encoded.sum(dim = (1, 2))

		miss_token = self.miss_proj(miss)
		return k0_token + k1_token + miss_token

	def forward(self, k0, k1, miss):
		z = self._encode_firm_token(k0, k1, miss)

		all_attn = []
		for block in self.blocks:
			z, attn_w = block(z)
			all_attn.append(attn_w)

		return self.head_3m(z), self.head_6m(z), self.head_12m(z), all_attn