import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from method import solve, calculate_distance_matrix

# 生成随机的3D点
num_points = 100
data = np.random.uniform(size=(num_points, 3)).astype(np.float32)
distance_matrix = calculate_distance_matrix(data)
route = solve(distance_matrix)

# 创建一个3D图形对象
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D散点图
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
# 初始点
ax.scatter(data[0, 0], data[0, 1], data[0, 2], c='r', s=100, marker='x')

# 绘制3D线图
for i in range(num_points):
    ax.text(data[i, 0], data[i, 1], data[i, 2], str(i))

# 绘制路径
colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_points))
for i in range(num_points):
    ax.plot([data[route[i], 0], data[route[i+1], 0]], 
            [data[route[i], 1], data[route[i+1], 1]], 
            [data[route[i], 2], data[route[i+1], 2]], c=colors[i])


# 设置标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
# plt.savefig('3d.png')
# print('3d.png saved')
