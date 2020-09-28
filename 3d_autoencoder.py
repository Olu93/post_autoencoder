# https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling/tree/master/Generative
# https://github.com/AnTao97/PointCloudDatasets/blob/master/dataset.py
# %%
import tensorflow as tf
import pickle
import pyvista as pv
from pyntcloud import PyntCloud
from tqdm import tqdm
import pandas as pd
import glob
import random
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import tensorflow.keras.optimizers as opt
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# %%
def retrieve_voxel_data(cloud, n=8):
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n, n_y=n, n_z=n)
    voxelgrid = cloud.structures[voxelgrid_id]
    vol = np.array(voxelgrid.get_feature_vector(mode="binary"), dtype=np.uint16)
    return vol, voxelgrid


def plot_binary_pointcound(cloud):
    vertices = np.argwhere(cloud)
    tcloud = PyntCloud(pd.DataFrame(vertices, columns="x y z".split()))
    tcloud.plot(mesh=True, width=400, height=400, backend="threejs")


# %%
# FROM PyVista
# original_point_cloud = pv.read("diamond.ply")
data_files = list(glob.glob(str("data/*.ply"), recursive=True))
num_meshes = len(data_files)
point_cloud_dataset_generator = (PyntCloud.from_file(mesh_file) for mesh_file in tqdm(data_files, total=num_meshes))
point_cloud_dataset_collected = list(point_cloud_dataset_generator)
f"Loaded point clouds for {num_meshes} meshes"
# %%
point_cloud_dataset_collected[1].plot(mesh=True, width=400, height=400, backend="threejs")
# %%
voxel_data_generator = (retrieve_voxel_data(cloud, 32) for cloud in tqdm(point_cloud_dataset_collected, total=num_meshes))
voxel_data_collected = list(voxel_data_generator)
voxel_data = np.array([item[0] for item in voxel_data_collected])
data = voxel_data.reshape(voxel_data.shape + (1, ))
data_shape = data.shape[1:]
f"Loaded voxel data for {num_meshes} meshes"
# %%
voxel_data_collected[1][1].plot(d=3, mode="density", width=400, height=400)
# %%
cut_point = .1
x_train, x_test = train_test_split(data.astype(np.float), shuffle=True, test_size=.1)
f"Training: {x_train.shape} | Test: {x_test.shape}"


# %%
def EncoderUnit(e, num_filters):
    e = layers.Conv3D(num_filters, 3, activation='elu', strides=2, padding='same', kernel_regularizer=reg.L1L2(.1, .1))(e)
    e = layers.Dropout(.5)(e)
    e = layers.BatchNormalization()(e)
    return e


def DecoderUnit(d, num_filters):
    d = layers.Conv3DTranspose(num_filters, 3, activation="elu", strides=2, padding="same", kernel_regularizer=reg.L1L2(.1, .1))(d)
    d = layers.Dropout(.5)(d)
    d = layers.BatchNormalization()(d)
    return d


def Encoder(data_shape, output_dim=8):
    encoder_inputs = tf.keras.Input(shape=data_shape)
    x = encoder_inputs
    x = EncoderUnit(x, 16)
    x = EncoderUnit(x, 32)
    x = EncoderUnit(x, 48)
    x = EncoderUnit(x, 64)
    unflattened_shape = list(x.shape[1:])
    x = layers.Flatten()(x)
    flattened_shape = list(x.shape[1:])[0]
    # e = layers.Dense(8, activation="elu")(e)
    z = layers.Dense(output_dim, kernel_regularizer=reg.L1L2(.1, .1))(x)
    # z = layers.Dropout(.5)(z)
    return Model(encoder_inputs, z), [output_dim, flattened_shape, unflattened_shape]


def Decoder(bridging_shapes):
    print(bridging_shapes)
    z = layers.Input(shape=(bridging_shapes[0], ))  # adapt this if using `channels_first` image data format

    d = layers.Dense(bridging_shapes[1], kernel_regularizer=reg.L1L2(.1, .1))(z)
    # d = layers.Dropout(.5)(d)
    d = layers.Reshape(bridging_shapes[2])(d)
    d = DecoderUnit(d, 64)
    d = DecoderUnit(d, 48)
    d = DecoderUnit(d, 32)
    d = DecoderUnit(d, 16)
    # d = layers.Conv3DTranspose(1, 3, activation="elu", strides=2, padding="same")(d)
    decoder_outputs = layers.Conv3DTranspose(1, 3, activation="sigmoid", strides=1, padding="same")(d)
    # decoder_outputs = tf.pow(decoder_outputs, 3)
    return Model(z, decoder_outputs)


def output_manipulation(x):
    return tf.where(tf.less_equal(x, .5), .0, x)


def generate_loss_function(penalty=.5):
    assert penalty >= 0 and penalty <= 1, "Choose value between 0 and 1"
    penalty = tf.constant(penalty, dtype=tf.float32)

    def loss_function(y, y_pred):
        # print(y_pred.shape)
        # print(y.shape)
        clipped_y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1.0 - 1e-7)
        binary_cross_entropy = -y * penalty * tf.math.log(clipped_y_pred) - (1 - penalty) * (1 - y) * tf.math.log(1 - clipped_y_pred)
        # print(binary_cross_entropy.shape)
        loss = tf.reduce_sum(tf.reduce_mean(binary_cross_entropy, axis=0))  #* tf.reshape(y_pred, (-1, )).shape[0]

        return loss

    def loss_function2(y, y_pred):
        mse = tf.reduce_mean(tf.pow(y - y_pred, 2))
        return mse

    return loss_function


def _rebase(x1, x2, min_val, max_val):
    x1 = (x1 * (max_val - min_val)) + min_val
    x2 = (x2 * (max_val - min_val)) + min_val
    return x1, x2


x = layers.Input(shape=data_shape)
# x = (x * tf.constant(6.0)) - tf.constant(1.0)
encoder, bridging_shapes = Encoder(data_shape, 300)
encoder.summary()
decoder = Decoder(bridging_shapes)
decoder.summary()

autoencoder = Model(x, decoder(encoder(x)))
autoencoder.summary()
## %%
loss_fn = generate_loss_function(0.97)
opt_cl = opt.SGD(learning_rate=0.001, nesterov=True, momentum=.9)
autoencoder.compile(optimizer='adam', loss=loss_fn)

# # x_train_mod =
x_train_mod, x_test_mod = _rebase(x_train, x_test, 0, 1)
y_train_mod, y_test_mod = _rebase(x_train, x_test, 0, 1)
# %%
autoencoder.fit(x_train_mod, y_train_mod, epochs=200, batch_size=1, shuffle=True, validation_data=(x_test_mod, y_test_mod))
# # %%
# plot_binary_pointcound(voxel_data_collected[1][0])
# %%
n = 5
test_sample = np.array(random.sample(list(x_train), n))
z = encoder.predict(test_sample)

originals = test_sample.reshape(test_sample.shape[:-1])
decodings = decoder.predict(z).reshape(test_sample.shape[:-1])
# %%
rows = 2
fig = plt.figure(figsize=(8, 5))
elem = None
for i in range(0, n):
    # display original
    cnt = i + 1
    ax = fig.add_subplot(rows, n, cnt, projection='3d')
    elem1 = np.argwhere(test_sample[i])
    ax.scatter(elem1[:, 0], elem1[:, 1], elem1[:, 2], cmap="Greys_r")
    encoding_strings = "\n".join([f"{feature:.2f}" for feature in z[i]])
    # ax = fig.add_subplot(rows, n, i + n)
    # ax.text(0.5, 0.5, encoding_strings, horizontalalignment='center', verticalalignment='center')
    # ax.axis('off')
    ax = fig.add_subplot(rows, n, cnt + 1 * n, projection='3d')
    # print(np.argwhere(decodings[i].reshape(data_shape[:-1]) > 0.5).shape)
    elem2 = np.argwhere(decodings[i] >= .9)
    ax.scatter(elem2[:, 0], elem2[:, 1], elem2[:, 2], cmap="Greys_r")

plt.show()


# %%
# Function to produce the data matrix for rendering.
def make_data_matrix(x, intensity):
    return intensity * np.repeat(np.repeat(np.repeat(x[0][0], 3, axis=0), 3, axis=1), 3, axis=2)


# %%
from mpl_toolkits.mplot3d import Axes3D
d_num = 1
fig = plt.figure()
ax = Axes3D(fig)
elem = np.argwhere(x_test[d_num].reshape(data_shape[:-1]))
ax.scatter(elem[:, 0], elem[:, 1], elem[:, 2])

# %%
fig = plt.figure()
ax = Axes3D(fig)
elem = np.argwhere(decodings[d_num].reshape(data_shape[:-1]) > 0.45)
ax.scatter(elem[:, 0], elem[:, 1], elem[:, 2])

# %%
tmp = PyntCloud(pd.DataFrame(elem, columns="x y z".split()))
tmp.plot(mesh=True, backend="threejs")
# %%
