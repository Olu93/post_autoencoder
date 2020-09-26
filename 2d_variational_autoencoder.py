# %% [markdown]
## Convolutional variational autoencoder for face generation!
# This code shows the example of an autoencoder applied on the MNIST and faces dataset.
# It is the natural extension of [./2d_autoencoder.ipynb](Part 1) of this series.
#
# These links complement the ones introduced in Part 1:
# - https://keras.io/examples/generative/vae/
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np

# %% [markdown]
# There's virtually no difference from part one. This just loads the MNIST dataset. Why MNIST, if this post is about face generations?
# As said in Part 1, MNIST is very often a good place to start! If it works with digits its likely to work with faces.
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

# %% [markdown]
# This is largely the same code as in Part 1. However, take a closer look at the encoder.
# Instead of returning an embedding vector only we return a mean and a log term of the variance vector.
# The logterm is more of a practical issue, as it ensures numerical stability when values get pretty small.
# Its not a very important detail, but if you want to know more, checkout this [https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/](comprehensive post)
# about VAE's and their implementation.
# Other than that we return a vector sample using the mean and variance. More on the sampling operation in the next cell.
# In the decoder, I swapped the fairly verbose Conv2D-Upsampling pattern with the Conv2DTranspose layer, because it does the same and is shorter, in principle.
# However, the devil lies in the detail. The Conv2DTranspose learns the best way to upsample the image.
# If you want to learn more about this layer checkout this [https://stackoverflow.com/a/53655426/4162265](StackOverflow-Answer)
# and this neat visual [https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d](Blog-Post explanation).
#


def Encoder(output_dim=8):
    encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
    e = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    e = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(e)
    unflattened_shape = list(e.shape[1:])
    e = layers.Flatten()(e)
    flattened_shape = list(e.shape[1:])[0]
    e = layers.Dense(16, activation="relu")(e)
    z_mean = layers.Dense(output_dim, name="z_mean")(e)
    z_log_var = layers.Dense(output_dim, name="z_log_var")(e)
    z = Sampling()([z_mean, z_log_var])
    return Model(encoder_inputs, [z_mean, z_log_var, z]), [output_dim, flattened_shape, unflattened_shape]


def Decoder(bridging_shapes):
    print(bridging_shapes)
    z = layers.Input(shape=(bridging_shapes[0], ))  # adapt this if using `channels_first` image data format

    d = layers.Dense(bridging_shapes[1])(z)
    d = layers.Reshape(bridging_shapes[2])(d)
    d = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(d)
    d = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(d)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(d)
    return Model(z, decoder_outputs)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
            reconstruction_loss *= np.prod(data.shape[1:])
            # reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction


# define input to the model:
x = layers.Input(shape=(28, 28, 1))
encoder, bridging_shapes = Encoder(2)
encoder.summary()
decoder = Decoder([2, 3136, [7, 7, 64]])
decoder.summary()
# make the model:
vae = VAE(encoder, decoder)

# %%
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=128)

# %%
encodings, z_mean, z_log_var = encoder.predict(x_test)
z_vars = np.array([np.eye(z_log_var.shape[1]) * var for var in np.exp(.5 * z_log_var)])
sampled_encodings = np.array([np.random.multivariate_normal(mean, variance) for mean, variance in zip(z_mean, z_vars)])
decodings = decoder.predict(encodings)

n = 10
rows = 3
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # display original
    ax = plt.subplot(rows, n, i)
    plt.imshow(x_test[i].reshape(28, 28), cmap="Greys_r")
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    encoding_strings = "\n".join([f"{feature:.2f}" for feature in encodings[i]])
    ax = plt.subplot(rows, n, i + n)
    ax.text(0.5, 0.5, encoding_strings, horizontalalignment='center', verticalalignment='center')
    ax.axis('off')
    # display reconstruction
    ax = plt.subplot(rows, n, i + 2 * n)
    plt.imshow(decodings[i].reshape(28, 28), cmap="Greys_r")
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# %%
# display a 2D manifold of the digits
# n = 30  # figure with 15x15 digits
# scale = 2
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # we will sample n points within [-15, 15] standard deviations
# grid_x = np.linspace(-scale, scale, n)
# grid_y = np.linspace(-scale, scale, n)[::-1]

# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = decoder.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit


# plt.figure(figsize=(10, 10))
# plt.xlabel("z[0]")
# plt.ylabel("z[1]")
# plt.imshow(figure, cmap="Greys_r")
# plt.show()
def plot_latent(encoder, decoder):
    # display a n*n 2D manifold of digits
    n = 30
    digit_size = 28
    scale = 2.0
    figsize = 15
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size, ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent(encoder, decoder)


# %%
def plot_label_clusters(encoder, decoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(encoder, decoder, x_train, y_train)
# %%
