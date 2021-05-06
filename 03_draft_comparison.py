import Data
import torch
import torch_geometric.utils as U
import torch.nn.parameter as P
import torch.nn.functional as F
import progressbar
from torch.nn import Linear
from torch_geometric.nn import GCNConv 

NUM_EPOCHS = 10

DS_LABEL = 'test_gen'
OUT_DIM = 1
NODE_FEATURES = 6
SUB_REP_DIM = 3

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
		self.MLP_pred  = MLP(32, 32, OUT_DIM)

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

graph = Data.getPyTorchGeoData(DS_LABEL)
print("Simple Skip Model")
print(graph)

ss_model     = SimpleSkip()
ss_optimizer = torch.optim.Adam(ss_model.parameters(), lr=0.01, weight_decay=5e-4)
ss_criterion = torch.nn.MSELoss()

def ss_train():
	ss_model.train()
	ss_optimizer.zero_grad()
	out = ss_model(graph.x, graph.edge_index)
	loss = ss_criterion(out[graph.train_mask], graph.y[graph.train_mask])
	loss.backward()
	ss_optimizer.step()
	return loss

def ss_test():
	ss_model.eval()
	out  = ss_model(graph.x, graph.edge_index)
	loss = ss_criterion(out[graph.test_mask], graph.y[graph.test_mask])

	# test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]
	# test_acc = int(test_correct.sum())/int(graph.test_mask.sum()) 

	return loss

for epoch in range(1, NUM_EPOCHS):
	loss = ss_train()
	print("ss - epoch {:0>} - {}".format(epoch, loss))


# Can't get a real 'prediction error' due to the memory issues
#  so I'll just look at the loss over the test-mask entries? idk. 
test_acc = ss_test()
print("ss - final test loss: {}".format(test_acc))
graph = Data.getPyTorchGeoData(DS_LABEL)
subreddit_map, activity_data = Data.getRedditData(DS_LABEL, graph.num_nodes)
ACT_SHAPE = (graph.num_nodes, len(subreddit_map))


class AggregateSubreddits(torch.nn.Module):
	def __init__(self, activity):
		super(AggregateSubreddits, self).__init__()

		self.S = activity
		self.R = P.Parameter(torch.rand((len(subreddit_map), SUB_REP_DIM)))

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
print("Reddit Aggregation Model")
rs_model = RedditSkip(activity_data)
print(rs_model)

rs_optimizer = torch.optim.Adam(rs_model.parameters(), lr=0.01, weight_decay=5e-4)
rs_criterion = torch.nn.MSELoss()

def rs_train():
	rs_model.train()
	rs_optimizer.zero_grad()
	out = rs_model(graph.x, graph.edge_index)
	loss = rs_criterion(out[graph.train_mask], graph.y[graph.train_mask])
	loss.backward()
	rs_optimizer.step()
	return loss

def rs_test():
	rs_model.eval()
	out  = rs_model(graph.x, graph.edge_index)
	loss = rs_criterion(out[graph.test_mask], graph.y[graph.test_mask])
	return loss

print("rs - training for {} epochs".format(NUM_EPOCHS))
for epoch in range(1, NUM_EPOCHS):
	loss = rs_train()
	print("rs - epoch {} - {}".format(epoch, loss))

test_acc = rs_test()
print("final test loss: {}".format(test_acc))