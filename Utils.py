import random
import torch
import math
import pathlib

def generateRandomSparseTensor(shape, density, max_value):
	num_items = math.floor(shape[0]*shape[1]*density)
	i, j, v = [], [], []
	for n in range(num_items):
		# Pick random u/v's in range
		i.append(random.randint(0, shape[0]-1))
		j.append(random.randint(0, shape[1]-1))
		v.append(random.randint(0, max_value))
	return torch.sparse_coo_tensor([i, j], v, shape, dtype=torch.float)


class Logger:
	def __init__(self, file_name):
		pathlib.Path(file_name).parent.mkdir(exist_ok=True)
		self.fn = file_name

	def addValue(self, v):
		with open(self.fn, 'a') as f:
			f.write("{}\n".format(v))

