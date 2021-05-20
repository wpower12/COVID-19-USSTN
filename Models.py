import torch
import torch_geometric.utils as U
import torch.nn.parameter as P
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

### Baseline Temporal Skip Model
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
		return out


class TemporalSkip(torch.nn.Module):
	def __init__(self, num_node_features, output_dim):
		super(TemporalSkip, self).__init__()

		# I ?think? these are values similar to what the paper uses.
		self.MLP_embed = MLP(64, num_node_features, 32)
		self.GCN1      = GCNConv(32, 32)
		self.GCN2      = GCNConv(32, 32)
		self.GCN3      = GCNConv(32, 32)
		self.MLP_pred  = MLP(32, 32, output_dim)
		
	def forward(self, x, edge_index, priors):
		# Initial Embedding
		h = self.MLP_embed(x)
		h = F.dropout(h, p=0.5, training=self.training)

		# First Hop
		h = self.GCN1(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()

		# Second Hop
		h = self.GCN2(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()
		
		# Third Hop
		h = self.GCN3(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()

		# Prediction layer
		out = self.MLP_pred(h)
		out = torch.add(out, priors)
		return out


class AggregateSubreddits(torch.nn.Module):
	def __init__(self, activity, sub_rep_dim):
		super(AggregateSubreddits, self).__init__()

		num_subs = len(activity[0])

		self.S = activity
		self.R = P.Parameter(torch.rand((num_subs, sub_rep_dim)))

	def forward(self, x):
		# Aggregate the info fromthe subreddit reps
		# weighted by activity
		sub_agg = torch.matmul(self.S, self.R)

		# Concatenate that with x features to be the
		# initial input to the model. 
		h = torch.cat((x, sub_agg), 1)
		return h


class RedditSkip(torch.nn.Module):
	def __init__(self, reddit_activity, num_node_features, sub_rep_dim, output_dim):
		super(RedditSkip, self).__init__()

		self.AggSubs   = AggregateSubreddits(reddit_activity, sub_rep_dim)
		
		self.MLP_embed = MLP(64, num_node_features+sub_rep_dim, 32)
		self.GCN1      = GCNConv(32, 32)
		self.GCN2      = GCNConv(32, 32)
		self.GCN3      = GCNConv(32, 32)
		self.MLP_pred  = MLP(32, 32, output_dim)

	def forward(self, x, edge_index, priors):

	
		# Initial Embedding from this 'subreddit updated'
		# initial representation. The rest is the same as 
		# the other model. 
		h = self.AggSubs(x)
		h = self.MLP_embed(h) # We use the Embedding MLP as our 'update'
		h = F.dropout(h, p=0.5, training=self.training)

		# First Hop
		h = self.GCN1(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()
		
		# Second Hop
		h = self.GCN2(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()

		# Third Hop
		h = self.GCN3(h, edge_index)
		h = F.dropout(h, p=0.5, training=self.training)
		h = h.relu()

		# Prediction layer
		out = self.MLP_pred(h)
		out = torch.add(out, priors)
		return out