import random
import torch
import math

def generateRandomSparseTensor(shape, density, max_value):
	num_items = math.floor(shape[0]*shape[1]*density)
	i, j, v = [], [], []
	for n in range(num_items):
		# Pick random u/v's in range
		i.append(random.randint(0, shape[0]-1))
		j.append(random.randint(0, shape[1]-1))
		v.append(random.randint(0, max_value))
	return torch.sparse_coo_tensor([i, j], v, shape, dtype=torch.float)
