import torch
import torch.nn as nn

## Time2Vec Temporal Encoding

class Time2Vec(nn.Module):
	def __init__(self, d_out):
		super().__init__()
		self.d_out = d_out
		self.omega = nn.Parameter(torch.randn(d_out))
		self.phi = nn.Parameter(torch.randn(d_out))

	def forward(self, lag_position):
		lag = lag_position.float().unsqueeze(-1)
		raw = self.omega * lag + self.phi
		out = torch.zeros_like(raw)
		out[..., 0] = raw[..., 0]
		out[..., 1:] = torch.sin(raw[..., 1:])
		return out

## Gated Residual Network

class GRN(nn.Module):
	def __init__(self, d_model, d_ff, dropout = 0.1):
		super().__init__()
		self.fc1 = nn.Linear(d_model, d_ff)
		self.fc2 = nn.Linear(d_ff, d_model * 2)
		self.layer_norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		residual = x
		h = F.elu(self.fc1(x))
		h = self.dropout(h)
		gated = self.fc2(h)
		value, gate = gated.chunk(2, dim = -1)
		h = value * torch.sigmoid(gate)
		return self.layer_norm(residual + h)

## Feature Encoding Varients

class LinearEncoder(nn.Module):
	def __init__(self, n_features, d_model):
		super().__init__()
		self.weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
		self.biases = nn.Parameter(torch.zeros(n_features, d_model))

	def forward(self, x):
		return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class PerFeatureTokeniser(nn.Module):
	def __init__(self, n_features, d_model):
		super().__init__()
		self.projections = nn.Parameter(torch.randn(n_features, 1, d_model) * 0.02)
		self.biases = nn.Parameter(torch.zeros(n_features, d_model))

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		proj = self.projections.squeeze(1).unsqueeze(0)
		return x_exp * proj + self.biases.unsqueeze(0)


class PiecewiseLinearEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_bins = 16):
		super().__init__()
		self.num_bins = num_bins
		boundaries = torch.linspace(-0.5, 0.5, num_bins + 1)
		self.register_buffer("boundaries", boundaries)
		self.feature_weights = nn.Parameter(torch.randn(n_features, num_bins, d_model) * 0.02)

	def _encode_bins(self, x):
		t_lower = self.boundaries[:-1]
		t_upper = self.boundaries[1:]
		x_exp = x.unsqueeze(-1)
		activations = torch.clamp((x_exp - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0)
		return activations

	def forward(self, x):
		bin_act = self._encode_bins(x)
		out = torch.einsum("bnk,nkd->bnd", bin_act, self.feature_weights)
		return out


class PeriodicEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq = 32):
		super().__init__()
		self.num_freq = num_freq
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.phi = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq, d_model)

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		sinusoidal = torch.sin(x_exp * self.omega.unsqueeze(0) + self.phi.unsqueeze(0))
		out = self.proj(sinusoidal)
		return out


def build_encoder(variant, n_features, d_model, ple_bins = 16, periodic_freq = 32):
	if variant == "linear":
		return LinearEncoder(n_features, d_model)
	elif variant == "per_feature":
		return PerFeatureTokeniser(n_features, d_model)
	elif variant == "ple":
		return PiecewiseLinearEncoder(n_features, d_model, num_bins = ple_bins)
	elif variant == "periodic":
		return PeriodicEncoder(n_features, d_model, num_freq = periodic_freq)
	else:
		raise ValueError(f"Unknown encoding variant: {variant}")