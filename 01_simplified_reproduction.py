import Data
import torch
import torch_geometric.utils as U
import torch.nn.functional as F
import progressbar
from torch.nn import Linear
from torch_geometric.nn import GCNConv 

DS_LABEL = 'test_gen'
OUT_DIM = 1
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
print(graph)

model     = SimpleSkip()
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

	# test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]
	# test_acc = int(test_correct.sum())/int(graph.test_mask.sum()) 

	return loss

for epoch in range(1, 2):
	loss = train()
	print("epoch {:0>} - {}".format(epoch, loss))


# Can't get a real 'prediction error' due to the memory issues
#  so I'll just look at the loss over the test-mask entries? idk. 
test_acc = test()
print("final test loss: {}".format(test_acc))