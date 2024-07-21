import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch


k = int(input('enter the amount of clusters: '))
data_size = -1

def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.show()


def calc_cost(V, C, Z, data):
    global k
    global data_size
    C_Z = torch.zeros(k * 100 if data_size == -1 else data_size, 2, dtype=torch.float32)
    for i in range(k * 100 if data_size == -1 else data_size):
        centroid_index = Z[i]
        C_Z[i] = C[:, int(centroid_index)]
    return (torch.norm(data - data @ V @ V.T) ** 2) + (torch.norm(data @ V - C_Z) ** 2)


def update_V(V, C, Z, data):
    V = V.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([V], lr=0.01)
    for iteration in range(300):
        optimizer.zero_grad()
        loss = calc_cost(V, C, Z, data)
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
    V, _ = torch.linalg.qr(V)
    return V.detach()


def update_C(V, C, Z, data):
    C = C.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([C], lr=0.01)
    for iteration in range(300):
        optimizer.zero_grad()  # Clear the gradients
        loss = calc_cost(V, C, Z, data)  # Compute the loss
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update C
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
    return C.detach()


def update_Z(V, C, Z, data):
    global data_size
    global k
    for i in range(k * 100 if data_size == -1 else data_size):
        min_error = 10000
        min_index = 0
        for j in range(k):
            res = (torch.norm(data[i] @ V - C[:, j]) ** 2)
            if res < min_error:
                min_error = res
                min_index = j
        Z[i] = min_index
    return Z


def train(V, C, Z, data, loops):
    prev_loss = -1
    for i in range(loops):
        loss = calc_cost(V, C, Z, data)
        if abs(loss - prev_loss) < 1:
            loops = i
            break
        prev_loss = loss
        print(f'iter {i + 1}, loss: {loss}')
        V = update_V(V, C, Z, data)
        C = update_C(V, C, Z, data)
        Z = update_Z(V, C, Z, data)
    return V, C, Z, loops


blobs_data, blobs_labels = make_blobs(n_samples=k*100, cluster_std=0.8, centers=k, n_features=3)#, random_state=0)

V = np.array([[1, 0], [0, 1], [0, 0]])
Z = np.zeros(k * 100)
C = np.random.rand(2, k) * 4 - 2

V = torch.tensor(V, dtype=torch.float32, requires_grad=True)
Z = torch.tensor(Z, dtype=torch.int64)
C = torch.tensor(C, dtype=torch.float32, requires_grad=True)
blobs_data = torch.tensor(blobs_data, dtype=torch.float32)

iters = int(input('iterations to train: '))

plot_clusters(blobs_data @ V.detach(), blobs_labels, 'correct labels for data')
plot_clusters(blobs_data @ V.detach(), Z, 'initial clustering')
V, C, Z, iters = train(V, C, Z, blobs_data, iters)
plot_clusters(blobs_data @ V.detach(), blobs_labels, f'{iters} training iterations wanted result')
plot_clusters(blobs_data @ V.detach(), Z, f'{iters} training iterations result')


# MNIST:
data_size = 4000
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=data_size,
                                           shuffle=True)

images, labels = next(iter(train_loader))
images = images.view(data_size, -1)

Z = np.zeros(data_size)
V = torch.zeros(784, 2)
V[0, 0] = 1
V[1, 1] = 1
V.requires_grad_()
C = np.random.rand(2, data_size) * 4 - 2
C = torch.tensor(C, dtype=torch.float32, requires_grad=True)

plot_clusters(images @ V.detach(), labels, 'correct labels for data')
plot_clusters(images @ V.detach(), Z, 'initial clustering')

k = 10
iters = int(input('iterations to train: '))
V, C, Z, iters = train(V, C, Z, images, iters)

plot_clusters(images @ V.detach(), labels, f'{iters} training iterations wanted result')
plot_clusters(images @ V.detach(), Z, f'{iters} training iterations result')
