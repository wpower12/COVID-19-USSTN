import torch
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import GCNConv 


NODE_FEATURES = 6


class MLP(torch.nn.Module):
	def __init__(self, num_hidden, dim_in, dim_out):
		super(MLP, self).__init__()
		self.H = num_hidden
		self.linear_1 = Linear(dim_in, self.H)
		self.linear_2 = Linear(self.H, dim_out)

	def forward(self, x):
		h = self.linear_1(x)
		h = h.tanh()
		h = self.linear_2(h)
		out = h.tanh()
		# Note: this was the soruce of my tuple issue. 
		#       used to be 'return h, out'
		return out


class SimpleSkip(torch.nn.Module):
	def __init__(self):
		super(SimpleSkip, self).__init__()

		# I think? these are values similar to what the paper uses.
		self.MLP_embed = MLP(64, NODE_FEATURES, 32)
		self.GCN1      = GCNConv(32, 32)
		self.GCN2      = GCNConv(32, 32)
		self.MLP_pred  = MLP(32, 2)

	def forward(self, x, edge_index):
		# Initial Embedding
		h = self.MLP_embed(x)

		# First Hop
		h = self.GCN1(h, edge_index)
		h = h.relu()
		h = F.dropout(h, p=0.5, training=self.training)
		# Second Hop
		h = self.GCN2(h, edge_index)
		h = h.relu()
		h = F.dropout(h, p=0.5, training=self.training)

		# Prediction layer
		out = self.MLP_pred(h)
		return out

