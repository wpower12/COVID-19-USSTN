import Data
import Utils
import torch
import torch.nn.parameter as P
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import GCNConv 

DS_LABEL = 'test_gen'
OUT_DIM = 1
NODE_FEATURES = 6

# This is just hardcoded now because I'm using synthetic 
# activity data. 
NUM_SUBREDDITS = 20000
SUB_REP_DIM = 3

graph    = Data.getPyTorchGeoData(DS_LABEL)
num_cdns = graph.num_nodes 
print(graph)

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


class AggregateSubreddits(torch.nn.Module):
	def __init__(self, activity):
		super(AggregateSubreddits, self).__init__()

		self.S = activity
		self.R = P.Parameter(torch.rand((NUM_SUBREDDITS, SUB_REP_DIM)))

	def forward(self, x):
		# Aggregate the info fromthe subreddit reps
		# weighted by activity
		sub_agg = torch.matmul(self.S, self.R)

		# Concatenate that with x features to be the
		# initial input to the model. 
		h = torch.cat((x, sub_agg), 1)
		return h


class RedditSkip(torch.nn.Module):
	def __init__(self, reddit_activity):
		super(RedditSkip, self).__init__()

		# I think? these are values similar to what the paper uses.
		self.MLP_embed = MLP(64, NODE_FEATURES+SUB_REP_DIM, 32)
		self.GCN1      = GCNConv(32, 32)
		self.GCN2      = GCNConv(32, 32)
		self.MLP_pred  = MLP(32, 32, OUT_DIM)
		self.AggSubs   = AggregateSubreddits(reddit_activity)

	def forward(self, x, edge_index):

	
		# Initial Embedding from this 'subreddit updated'
		# initial representation. The rest is the same as 
		# the other model. 
		h = self.AggSubs(x)
		h = self.MLP_embed(h) # We use the Embedding MLP as our 'update'

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

# We'll use fake activity data for now, but this is about the 
# same density.
ACT_SHAPE = (num_cdns, NUM_SUBREDDITS)
fake_rd = Utils.generateRandomSparseTensor(ACT_SHAPE, 0.001, 5)
model = RedditSkip(fake_rd)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

def train():
	model.train()
	optimizer.zero_grad()
	out = model(graph.x, graph.edge_index)
	loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
	loss.backward()
	optimizer.step()
	return loss

def test():
	model.eval()
	out  = model(graph.x, graph.edge_index)
	loss = criterion(out[graph.test_mask], graph.y[graph.test_mask])
	return loss

for epoch in range(1, 5):
	loss = train()
	print("epoch {} - {}".format(epoch, loss))

test_acc = test()
print("final test loss: {}".format(test_acc))