import numpy as np

# 随机生成数据
a = np.random.randn(8, 10, 3)
b = np.random.randn(8, 10, 3)

# 使用 einsum 进行矩阵乘法
result = a @ b.transpose(0, 2, 1)

print(result.shape)  # 结果形状应该是 (8, 10, 10)

# 使用 numpy 进行矩阵乘法
# result1 = np.matmul(result, a)
result1 = result @ a

print(result1.shape)  # 结果形状应该是 (8, 10, 3)
