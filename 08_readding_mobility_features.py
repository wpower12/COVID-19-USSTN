import Data
import Utils
import Models

import torch
import torch_geometric.utils as U
import torch.nn.parameter as P
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv 

DS_LABEL = 'w7_wDeltas'
RES_LABEL = 'w7_wDeltas_00'

RES_DIR  = "results/{}".format(RES_LABEL)
CHECKPOINT_FN = "{}/{}".format(RES_DIR, "checkpoint.pt")

NUM_EPOCHS = 1000000
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 5e-4
REPORT_EVERY = 1
LOSS_BUFF_SIZE = 1
OUT_DIM = 1
NODE_FEATURES = 48
SUB_REP_DIM = 3
CUDA_CORE = 0

dev_str = "cuda:{}".format(CUDA_CORE)
device = torch.device(dev_str if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

print("Using dataset: {}".format(DS_LABEL))
graph = Data.getPyTorchGeoData(DS_LABEL)
print(graph)

print("Temporal Skip Model:")
ts_model = Models.TemporalSkip(NODE_FEATURES, OUT_DIM)
ts_model.to(device)
print(ts_model)

ts_log_fn = "{}/{}".format(RES_DIR, "ts_loss_per_epoch.txt")
print("saving results to: {}".format(ts_log_fn))
ts_log = Utils.Logger(ts_log_fn)

ts_optimizer = torch.optim.Adam(ts_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
ts_criterion = Utils.RMSLELoss(reduction='mean')

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


def saveCheckpoint(path, model, optim, epoch, loss):
	save_dict = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optim.state_dict(),
		'loss': loss 
	}
	torch.save(save_dict, path)


loss_buffer = []
for epoch in range(1, NUM_EPOCHS):
	loss = ts_train()
	if epoch % REPORT_EVERY == 0:
		print("ts - epoch {:0>} - {}".format(epoch, loss))

	loss_buffer.append(loss)
	if len(loss_buffer) > LOSS_BUFF_SIZE:
		ts_log.addValues(loss_buffer)
		loss_buffer = []
		# We'll also use this time to update the savepoint.
		saveCheckpoint(CHECKPOINT_FN, ts_model, ts_optimizer, epoch, loss)

