import Data
import Utils
import Models

import torch

DS_LABEL = 'w7_readded_mob'
RES_LABEL = 'testing_00_skinny_draft'

RES_DIR  = "results/{}".format(RES_LABEL)
CHECKPOINT_FN = "{}/{}".format(RES_DIR, "checkpoint.pt")

NUM_EPOCHS = 1000000
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 5e-4
REPORT_EVERY = 100
LOSS_BUFF_SIZE = 100
OUT_DIM = 1
NODE_FEATURES = 48
SUB_REP_DIM = 3
CUDA_CORE = 0

HEIGHT = 5
WIDTH  = 16

dev_str = "cuda:{}".format(CUDA_CORE)
device = torch.device(dev_str if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

print("Using dataset: {}".format(DS_LABEL))
graph = Data.getPyTorchGeoData(DS_LABEL)
print(graph)

model = Models.SkinnySkip(NODE_FEATURES, HEIGHT, WIDTH, OUT_DIM)
print(model)

log_fn = "{}/{}".format(RES_DIR, "loss_per_epoch.txt")
log = Utils.Logger(log_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = Utils.RMSLELoss(reduction='mean')

print("saving results to: {}".format(log_fn))


def train():
	model.train()
	optimizer.zero_grad()
	out  = model(graph.x.to(device), graph.edge_index.to(device), graph.priors.to(device))
	loss = criterion(out[graph.train_mask].to(device), graph.y[graph.train_mask].to(device))
	loss.backward()
	optimizer.step()
	return loss


def test():
	model.eval()
	out  = model(graph.x.to(device), graph.edge_index.to(device), graph.priors.to(device))
	loss = criterion(out[graph.test_mask].to(device), graph.y[graph.test_mask].to(device))
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
	loss = train()
	if epoch % REPORT_EVERY == 0:
		print("epoch {:0>} - {}".format(epoch, loss))

	loss_buffer.append(loss)
	if len(loss_buffer) > LOSS_BUFF_SIZE:
		log.addValues(loss_buffer)
		loss_buffer = []
		# We'll also use this time to update the savepoint.
		saveCheckpoint(CHECKPOINT_FN, model, optimizer, epoch, loss)

