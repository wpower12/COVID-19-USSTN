import random
import torch
import math
import pandas as pd
import pathlib


def getDateRange(start, end):
	START_DATE  = pd.to_datetime(start)
	END_DATE    = pd.to_datetime(end)
	return pd.date_range(start=START_DATE, end=END_DATE, freq='D')


def generateRandomSparseTensor(shape, density, max_value):
	num_items = math.floor(shape[0]*shape[1]*density)
	i, j, v = [], [], []
	for n in range(num_items):
		# Pick random u/v's in range
		i.append(random.randint(0, shape[0]-1))
		j.append(random.randint(0, shape[1]-1))
		v.append(random.randint(0, max_value))
	return torch.sparse_coo_tensor([i, j], v, shape, dtype=torch.float)


def writeMapToCSV(fn, src_map, headers):
	pathlib.Path(fn).parent.mkdir(exist_ok=True)

	# doing this manually bc i cant get pandas to do it right?
	# i mean its def me but lets just say its the panda.
	with open(fn, 'w') as f:
		f.write("{}\n".format(",".join(headers)))
		for key in src_map:
			val = src_map[key]
			f.write("{}, {}\n".format(key, val))


def writeListToCSV(fn, src_list):
	pathlib.Path(fn).parent.mkdir(exist_ok=True)
	save_df = pd.DataFrame(src_list)
	save_df.to_csv(fn, header=False, index=False)
	

class Logger:
	def __init__(self, file_name):
		pathlib.Path(file_name).parent.mkdir(exist_ok=True)
		self.fn = file_name

	def addValue(self, v):
		with open(self.fn, 'a') as f:
			f.write("{}\n".format(v))

	def addValues(self, vs):
		with open(self.fn, 'a') as f:
			for v in vs:
				f.write("{}\n".format(v))


class RMSLELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


