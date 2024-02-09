import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


# Updating Discriminator
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """
    Update Discriminator
    
    Params
    ------
    X: Input of Discriminator
    Z: Input of Generator
    net_D: Discrimiator Network
    net_G: Generator Network
    loss: loss
    trainer_D: trainer object
    """
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape))+loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D


# Updating Generator
def update_G(Z, net_D, net_G, loss, trainer_G):
    """
    Updating Generator

    Params
    ------
    Z: Input of generator
    net_D: Discrimiator Network
    net_G: Generator Network
    loss: loss
    trainer_G: trainer object
    """
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G


def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0,0.02)

    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    
    for epoch in range(num_epochs):
        loss_d_accumulator = []
        loss_g_accumulator = []
        for X in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            loss_D = update_D(X, Z, net_D, net_G, loss, trainer_D)
            loss_G = update_G(Z, net_D, net_G, loss, trainer_G)
            loss_d_accumulator.append(loss_D.clone().detach().numpy())
            loss_g_accumulator.append(loss_G.clone().detach().numpy())
        print(f"Epoch: {epoch} | mean loss_D: {np.mean(loss_d_accumulator)} | mean loss_G: {np.mean(loss_g_accumulator)}")

        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        plt.scatter(data[:, 0], data[:, 1], c="r", label="real")
        plt.scatter(fake_X[:, 0], fake_X[:, 1], c="b", label="fake")
        plt.savefig(f"epoch_{epoch}_output.jpg")


if __name__ == "__main__":

    # Create Dataset
    X = torch.normal(0.0, 1, (1000, 2))
    A = torch.tensor([[1,2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X,A) + b
    plt.scatter(data[:100, (0)].detach().numpy(), data[:100, (1)].detach().numpy())
    plt.savefig("torch_synthetic_data.jpg")
    print(f"\nThe covariance of matrix is: \n{torch.matmul(A.T, A)}")


    # Training parameters
    batch_size = 8
    lr_D = 0.05
    lr_G = 0.005
    latent_dim = 2
    num_epochs = 20
    

    # Generator network
    net_G = nn.Sequential(nn.Linear(2,2))
    print(f"\nGenerator network: \n{net_G}")


    # Discriminator Network
    net_D = nn.Sequential(
        nn.Linear(2, 5),
        nn.Tanh(),
        nn.Linear(5, 3),
        nn.Tanh(),
        nn.Linear(3, 1)
    )
    print(f"\nDiscriminator network: \n{net_D}")


    # Create Dataloader
    data_iter = torch.utils.data.DataLoader(data, batch_size=batch_size)


    # Call Training
    train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data[:100].detach().numpy())