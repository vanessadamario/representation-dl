from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


min_n = 8; max_n = 200
min_p = 10; max_p = 100
n_points = 16; repetition = 20


sample_dim = np.arange(min_n, max_n, n_points)
feature_dim = np.arange(min_p, max_p, n_points)


results = np.load("poly_results.npy")
new_results = np.zeros((sample_dim.size, feature_dim.size, repetition, 2))

for i in range(sample_dim.size):
    for j in range(feature_dim.size):
        new_results[i, j, :, :] = results[i, j, :, :]

accuracy = np.mean(new_results[:, :, :, 0], axis=-1)
ravel_accuracy = np.array([])
ravel_n = np.array([])
ravel_p = np.array([])

for i in range(sample_dim.size):
    for j in range(feature_dim.size):
        ravel_n = np.append(ravel_n, sample_dim[i])
    ravel_p = np.append(ravel_p, feature_dim)
    ravel_accuracy = np.append(ravel_accuracy, accuracy[i, :])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ravel_n, ravel_p, ravel_accuracy, marker='o')
ax.set_xlabel('# training samples')
ax.set_ylabel('# features')
ax.set_zlabel('accuracy score')

plt.show()
plt.close()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n_matrix = np.zeros((sample_dim.size, feature_dim.size))
# p_matrix = np.zeros((sample_dim.size, feature_dim.size))
# for i in range(sample_dim.size):
#     n_matrix[i, :] = np.repeat(sample_dim[i], feature_dim.size)
#     p_matrix[i, :] = feature_dim
# ax.plot_surface(n_matrix, p_matrix, accuracy)
# ax.set_xlabel('# training samples')
# ax.set_ylabel('# features')
# ax.set_zlabel('accuracy score')
#
# plt.show()
# plt.close()
