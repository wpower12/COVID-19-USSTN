import Data
import Utils
import torch
import torch_geometric.utils as U
import torch.nn.parameter as P
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv 

DS_LABEL = 'w7_copy'
RES_DIR  = "results/{}".format(DS_LABEL)
NUM_EPOCHS = 1000000
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 5e-4
REPORT_EVERY = 500
LOSS_BUFF_SIZE = 100
OUT_DIM = 1
NODE_FEATURES = 42
SUB_REP_DIM = 3
CUDA_CORE = 0

dev_str = "cuda:{}".format(CUDA_CORE)
device = torch.device(dev_str if torch.cuda.is_available() else "cpu")

print("Using dataset: {}".format(DS_LABEL))
graph = Data.getPyTorchGeoData(DS_LABEL)
print(graph)


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
	def __init__(self):
		super(TemporalSkip, self).__init__()

		# I ?think? these are values similar to what the paper uses.
		self.MLP_embed = MLP(64, NODE_FEATURES, 32)
		self.GCN1      = GCNConv(32, 32)
		self.GCN2      = GCNConv(32, 32)
		self.MLP_pred  = MLP(32, 32, OUT_DIM)
		
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

		# Prediction layer
		out = self.MLP_pred(h)
		out = torch.add(out, priors)
		return out


class RMSLELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


print("Temporal Skip Model:")
ts_model = TemporalSkip()
ts_model.to(device)

print(ts_model)
ts_log_fn = "{}/{}".format(RES_DIR, "ts_loss_per_epoch.txt")
print("saving results to: {}".format(ts_log_fn))
ts_log = Utils.Logger(ts_log_fn)

ts_optimizer = torch.optim.Adam(ts_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
ts_criterion = RMSLELoss(reduction='mean')

def ts_train():
	ts_model.train()
	ts_optimizer.zero_grad()
	out  = ts_model(graph.x.to(device), graph.edge_index.to(device), graph.priors.to(device))
	loss = ts_criterion(out[graph.train_mask].to(device), graph.y[graph.train_mask].to(device))
	loss.backward()
	ts_optimizer.step()
	return loss

def ts_test():
	ts_model.eval()
	out  = ts_model(graph.x.to(device), graph.edge_index.to(device), graph.priors.to(device))
	loss = ts_criterion(out[graph.test_mask].to(device), graph.y[graph.test_mask].to(device))
	return loss

loss_buffer = []
for epoch in range(1, NUM_EPOCHS):
	loss = ts_train()
	if epoch % REPORT_EVERY == 0:
		print("ts - epoch {:0>} - {}".format(epoch, loss))

	loss_buffer.append(loss)
	if len(loss_buffer) > LOSS_BUFF_SIZE:
		ts_log.addValues(loss_buffer)
		loss_buffer = []


test_acc = ts_test()
print("ts - final test loss: {}".format(test_acc))


### Reddit Assitted Model
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

		self.AggSubs   = AggregateSubreddits(reddit_activity)
		
		# I think? these are values similar to what the paper uses.

		self.MLP_embed = MLP(64, NODE_FEATURES+SUB_REP_DIM, 32)
		self.GCN1      = GCNConv(32, 32)
		self.GCN2      = GCNConv(32, 32)
		self.MLP_pred  = MLP(32, 32, OUT_DIM)

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

		# Prediction layer
		out = self.MLP_pred(h)
		out = torch.add(out, priors)
		return out


print("Reddit Aggregation Model")
rs_model = RedditSkip(activity_data.to(device))
rs_model.to(device)

print(rs_model)
rs_log_fn = "{}/{}".format(RES_DIR, "rs_loss_per_epoch.txt")
print("saving results to: {}".format(rs_log_fn))
rs_log = Utils.Logger(rs_log_fn)

rs_optimizer = torch.optim.Adam(rs_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# rs_criterion = torch.nn.MSELoss(reduction='mean')
rs_criterion = RMSLELoss(reduction='mean')

def rs_train():
	rs_model.train()
	rs_optimizer.zero_grad()
	out = rs_model(graph.x.to(device), graph.edge_index.to(device), graph.priors.to(device))
	loss = rs_criterion(out[graph.train_mask].to(device), graph.y[graph.train_mask].to(device))
	loss.backward()
	rs_optimizer.step()
	return loss

def rs_test():
	rs_model.eval()
	out  = rs_model(graph.x.to(device), graph.edge_index.to(device), graph.priors.to(device))
	loss = rs_criterion(out[graph.test_mask].to(device), graph.y[graph.test_mask].to(device))
	return loss

print("rs - training for {} epochs".format(NUM_EPOCHS))
loss_buffer = []
for epoch in range(1, NUM_EPOCHS):
	loss = rs_train()
	rs_log.addValue(loss)

	if epoch % REPORT_EVERY == 0:
		print("rs - epoch {} - {}".format(epoch, loss))

	loss_buffer.append(loss)
	if len(loss_buffer) > LOSS_BUFF_SIZE:
		rs_log.addValues(loss_buffer)
		loss_buffer = []

test_acc = rs_test()
print("final test loss: {}".format(test_acc))