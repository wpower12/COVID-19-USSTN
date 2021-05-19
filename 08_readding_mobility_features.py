import Data

START_DATE  = "04/12/2020"
END_DATE    = "12/31/2020"
TRAIN_SPLIT_IDX  = 200 # Leaves 64 to test
WINDOW_SIZE = 7 
DS_LABEL = 'w7_readded_mob'
RES_DIR  = "results/{}".format(DS_LABEL)
NUM_EPOCHS = 1000000
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 5e-4
REPORT_EVERY = 100
LOSS_BUFF_SIZE = 100
OUT_DIM = 1
NODE_FEATURES = 48
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
		self.GCN3      = GCNConv(32, 32)
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
		
		# Third Hop
		h = self.GCN3(h, edge_index)
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
