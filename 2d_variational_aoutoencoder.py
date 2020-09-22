# %%
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# %%
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


# %%
def Encoder(output_dim=8):
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    e = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    flat_e = Flatten()(e)
    mean_e = Dense(output_dim)(flat_e)
    std_e = Dense(output_dim)(flat_e)
    random_tensor = tf.random.normal((output_dim,))
    sampled_vector = (mean_e + std_e) * random_tensor
    return Model(input_img, sampled_vector), [output_dim, flat_e.shape[1], e.shape]


def Decoder(bridging_shapes):
    sampled_vector = Input(shape=(bridging_shapes[0], ))  # adapt this if using `channels_first` image data format

    d = Dense(bridging_shapes[1])(sampled_vector)
    d = tf.reshape(d, [-1] + list(bridging_shapes[2][1:]))
    d = Conv2D(8, (3, 3), activation='relu', padding='same')(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(8, (3, 3), activation='relu', padding='same')(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(16, (3, 3), activation='relu')(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)
    return Model(sampled_vector, d)


# define input to the model:
x = Input(shape=(28, 28, 1))
encoder, bridging_shapes = Encoder(2)
# encoder.summary()
decoder = Decoder(bridging_shapes)
# decoder.summary()
# make the model:
autoencoder = Model(x, decoder(encoder(x)))

# compile the model:
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=16, shuffle=True, validation_data=(x_test, x_test))

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

    encoding_strings = "\n".join([f"{feature:.2f}" for feature in encodings[i]])
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
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
# %%
