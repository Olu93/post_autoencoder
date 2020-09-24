# https://keras.io/examples/generative/vae/
# %%
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

# %%

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


# %%
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def Encoder(output_dim=8):
    encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    return_shape = x.shape
    flat_e = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(flat_e)
    z_mean = layers.Dense(output_dim, name="z_mean")(x)
    z_log_var = layers.Dense(output_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(encoder_inputs, [z, z_mean, z_log_var]), [output_dim, flat_e.shape[1], return_shape]


def Decoder(bridging_shapes):
    z = layers.Input(shape=(bridging_shapes[0], ))  # adapt this if using `channels_first` image data format

    d = layers.Dense(bridging_shapes[1])(z)
    d = tf.reshape(d, [-1] + list(bridging_shapes[2][1:]))
    d = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(d)
    d = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(d)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(d)
    return Model(z, decoder_outputs)


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
decoder = Decoder(bridging_shapes)
decoder.summary()
# make the model:
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=32)


# %%
encodings = encoder.predict(x_test)
decodings = decoder.predict(encodings)

n = 10
rows = 3
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # display original
    ax = plt.subplot(rows, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    encoding_strings = "\n".join([f"{feature:.2f}" for feature in encodings[0][i]])
    ax = plt.subplot(rows, n, i + n)
    ax.text(0.5, 0.5, encoding_strings, horizontalalignment='center', verticalalignment='center')
    ax.axis('off')
    # display reconstruction
    ax = plt.subplot(rows, n, i + 2 * n)
    plt.imshow(decodings[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# %%
# display a 2D manifold of the digits
n = 30  # figure with 15x15 digits
scale = 4
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-scale, scale, n)
grid_y = np.linspace(-scale, scale, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

# %%
