import numpy as np
from method import normalize
data = np.random.uniform(size=(5, 2)).astype(np.float32)



mean = data.mean(axis=0, keepdims=True)
std = data.std(axis=0, keepdims=True)

print(data)
print(mean)
print(std)
data_norm = (data - mean) / std
print(data_norm)

print(data_norm.mean(axis=0, keepdims=True))
print(data_norm.std(axis=0, keepdims=True))