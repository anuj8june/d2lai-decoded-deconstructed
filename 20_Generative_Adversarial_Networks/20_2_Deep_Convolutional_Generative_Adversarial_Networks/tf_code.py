import os
import math
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib.request import urlretrieve


def transform_func(X):
    X = X/255.
    X = (X-0.5)/0.5
    return X


class G_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kerne_size=4, strides=2, padding="same", *args, **kwargs):
        super().__init__(**kwargs)
        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(out_channels, kerne_size, strides, padding)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, X):
        return self.activation(self.batch_norm(self.conv2d_transpose(X)))
    

class D_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kerne_size=4, strides=2, padding="same", alpha=0.2, *args, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(out_channels, kerne_size, strides, padding)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha)
    
    def call(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


n_G = 64    
net_G = tf.keras.Sequential([
    G_block(out_channels=n_G*8, strides=1, padding="valid"),
    G_block(out_channels=n_G*4),
    G_block(out_channels=n_G*2),
    G_block(out_channels=n_G),
    tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh")
])

n_D = 64
net_D = tf.keras.Sequential([
    D_block(n_D),
    D_block(out_channels=n_D*2),
    D_block(out_channels=n_D*4),
    D_block(out_channels=n_D*8),
    tf.keras.layers.Conv2D(1, kernel_size=4, use_bias=False)
])


def update_G(Z, net_D, net_G, loss, optimizer_G):
    """
    Update Generator

    Params
    ------
    Z: Input to generator
    net_D: Discriminator Network
    net_G: Generator Network
    loss: loss function
    optimizer_G: optimizer for generator

    Returns
    -------
    loss_G: loss for generator
    """
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        fake_x = net_G(Z)
        fake_y = net_D(fake_x)
        loss_G = loss(ones, tf.squeeze(fake_y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G


def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    """
    Update Discriminator

    Params
    ------
    X: Input to Discriminator
    Z: Input to generator
    net_D: Discriminator Network
    net_G: Generator Network
    loss: loss function
    optimizer_D: optimizer for discriminator

    Returns
    -------
    loss_D: loss for discriminator
    """
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    zeros = tf.zeros((batch_size,))
    fake_x = net_G(Z)
    with tf.GradientTape() as tape:
        real_y = net_D(X)
        fake_y = net_D(fake_x)
        loss_D = (loss(ones, tf.squeeze(real_y)) + loss(zeros, tf.squeeze(fake_y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))

    optimizer_hp = {"learning_rate": lr, "beta_1": 0.5, "beta_2": 0.999}
    optimizer_G = tf.keras.optimizers.Adam(**optimizer_hp)
    optimizer_D = tf.keras.optimizers.Adam(**optimizer_hp)

    for epoch in range(1, num_epochs+1):
        loss_d_accumulator = []
        loss_g_accumulator = []
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(mean=0, stddev=1, shape=(batch_size, 1, 1, latent_dim))
            loss_D = update_D(X, Z, net_D, net_G, loss, optimizer_D)
            loss_G = update_G(Z, net_D, net_G, loss, optimizer_G)
            loss_d_accumulator.append(loss_D)
            loss_g_accumulator.append(loss_G)
        print(f"Epoch: {epoch} | mean loss_D: {np.mean(loss_d_accumulator)} | mean loss_G: {np.mean(loss_g_accumulator)}")

        # Show generated examples per epoch
        Z = tf.random.normal(mean=0, stddev=1, shape=(21, 1, 1, latent_dim))
        fake_x = net_G(Z) / 2 + 0.5
        _, axs = plt.subplots(7, 3, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(fake_x, axs):
            ax.imshow(img)
        plt.savefig(f"pytorch_epoch_{epoch}_generated_samples.jpg")
        


if __name__ == "__main__":
    # Download data and extract
    zip_path = "pokemon.zip"
    extracted_path = "./"
    data_dir = "./pokemon"
    data_url = "http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip"

    if not os.path.exists(zip_path):
        msg, _ = urlretrieve(data_url, zip_path)
        print(f"\nZip file successfully download in directory: {msg}")
    else:
        print(f"\nZip file already exists in directory: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
        print(f"\nFile extracted successfully in : {extracted_path}")

    
    # Training params
    batch_size = 256
    latent_dim = 100
    lr = 0.0005
    epochs = 40

    # Dataloader
    pokemon = tf.keras.preprocessing.image_dataset_from_directory(data_dir, batch_size=batch_size, image_size=(64,64))
    data_iter = pokemon.map(lambda x,y: (transform_func(x),y), num_parallel_calls=tf.data.AUTOTUNE)
    data_iter = data_iter.cache().shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Plot 20 image examples from dataset
    num_sample_plots = 20
    _, axs = plt.subplots(4, 5, figsize=(12, 12))
    for X,y in data_iter.take(1):
        imgs = X[:num_sample_plots,:,:,:]
        axs = axs.flatten()
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
    plt.savefig("tf_data_sample.jpg")

    # Testing net_G and net_D forward pass
    X = tf.zeros((1,1,1,100))
    print(f"Generator model forward pass: {net_G(X).shape}")

    X = tf.zeros((1,64,64,3))
    print(f"Discriminator model forward pass: {net_D(X).shape}")

    # Call training function
    train(net_D, net_G, data_iter, epochs, lr, latent_dim)