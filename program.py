import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

k = 3  # int(input('enter the amount of clusters: '))


def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.show()


def calc_cost(V, C, Z, data):
    global k
    C_Z = np.zeros((k * 100, 2))
    for i in range(k * 100):
        centroid_index = Z[i]
        C_Z[i] = C[:, int(centroid_index)]
    return (np.linalg.norm(data - data @ V @ V.T) ** 2) + (np.linalg.norm(data @ V - C_Z) ** 2)


def update_V(V, C, Z, data):
    return V


def update_C(V, C, Z, data):
    return C


def update_Z(V, C, Z, data):
    for i in range(k * 100):
        min_error = 10000
        min_index = 0
        for j in range(k):
            res = (np.linalg.norm(data[i] @ V - C[:, j]) ** 2)
            if res < min_error:
                min_error = res
                min_index = j
        Z[i] = min_index
    return Z


def train(V, C, Z, data, loops):
    for i in range(loops):
        print(f'iter {i + 1}, loss: {calc_cost(V, C, Z, data)}')
        V = update_V(V, C, Z, data)
        C = update_C(V, C, Z, data)
        Z = update_Z(V, C, Z, data)
    return V, C, Z


blobs_data, blobs_labels = make_blobs(n_samples=k * 100, cluster_std=0.8, centers=k, n_features=3, random_state=1)

V = np.array([[1, 0],
              [0, 1],
              [0, 0]])
Z = np.zeros(k * 100)
C = np.random.rand(2, k) * 4 - 2

V, C, Z = train(V, C, Z, blobs_data, 10)
plot_clusters(blobs_data @ V, blobs_labels, 'correct labels for data')
plot_clusters(blobs_data @ V, Z, 'estimated labels for data')
