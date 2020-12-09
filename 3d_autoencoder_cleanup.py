# https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling/tree/master/Generative
# https://github.com/AnTao97/PointCloudDatasets/blob/master/dataset.py
# %%
from datetime import datetime
from enum import auto
import functools
import io
from pyntcloud.structures.voxelgrid import VoxelGrid
import tensorflow as tf
import pickle
import pyvista as pv
from pyntcloud import PyntCloud
from tensorflow.python.keras.engine.base_layer import Layer
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
import pathlib
from matplotlib.backends.backend_agg import FigureCanvasAgg

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.run_functions_eagerly(True)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# %%
def compose(f, g):
    return lambda arg: f(g(arg))


def retrieve_voxel_data(cloud, n=8):
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n, n_y=n, n_z=n)
    voxelgrid = cloud.structures[voxelgrid_id]
    vol = np.array(voxelgrid.get_feature_vector(mode="binary"), dtype=np.uint16)
    return vol, voxelgrid


def plot_binary_pointcound(vertex_list):
    vertices = np.argwhere(vertex_list)
    tcloud = PyntCloud(pd.DataFrame(vertices, columns="x y z".split()))
    tcloud.plot(mesh=True, width=400, height=400, backend="threejs")


def generate_img_of_decodings(encodings, decodings, n=5, thresh=.8):
    rows = 2
    fig = plt.figure(figsize=(8, 5))

    for i in range(0, n):
        cnt = i + 1
        ax = fig.add_subplot(rows, n, cnt, projection='3d')
        elem1 = np.argwhere(encodings[i])
        ax.scatter(elem1[:, 0], elem1[:, 1], elem1[:, 2], cmap="Greys_r")
        # encoding_strings = "\n".join([f"{feature:.2f}" for feature in z[i]])
        ax = fig.add_subplot(rows, n, cnt + 1 * n, projection='3d')
        elem2 = np.argwhere(decodings[i] >= thresh)
        ax.scatter(elem2[:, 0], elem2[:, 1], elem2[:, 2], cmap="Greys_r")

    return fig


def generate_img_of_decodings_expanded(encodings, decodings, thresholds=[.9, .95, .99, .999, 1]):
    rows = 1
    fig = plt.figure(figsize=(25, 5))
    n = len(thresholds) + 1

    ax = fig.add_subplot(rows, n, 1, projection='3d')
    elem1 = np.argwhere(encodings[0])
    ax.set_title("Original")
    ax.scatter(elem1[:, 0], elem1[:, 1], elem1[:, 2], cmap="Greys_r")
    for i, tresh in enumerate(thresholds):
        cnt = i + 1
        ax = fig.add_subplot(rows, n, cnt + 1, projection='3d')
        ax.set_title(f"> {tresh:05}")
        elem2 = np.argwhere(decodings[0] >= tresh)
        ax.scatter(elem2[:, 0], elem2[:, 1], elem2[:, 2], cmap="Greys_r")

    return fig


def plot_pointcound(pynt_cloud_object, n=32):
    vol, example_voxelgrid = retrieve_voxel_data(pynt_cloud_object, n)
    return example_voxelgrid.plot(d=3, mode="binary", cmap="hsv", width=400, height=400)


# %%
path_to_dataset = pathlib.Path("data/dataset.pkl")
point_cloud_dataset_collected = None
num_meshes = None
if path_to_dataset.exists():
    point_cloud_dataset_collected = pickle.load(io.open(path_to_dataset, "rb"))
    num_meshes = len(point_cloud_dataset_collected)

if not path_to_dataset.exists():
    data_files = list(glob.glob(str("data/**/*.ply"), recursive=True))
    num_meshes = len(data_files)
    point_cloud_dataset_generator = (PyntCloud.from_file(mesh_file) for mesh_file in tqdm(data_files, total=num_meshes))
    point_cloud_dataset_collected = list(point_cloud_dataset_generator)
    pickle.dump(file=io.open(path_to_dataset, "wb"), obj=point_cloud_dataset_collected)
    print(f"Loaded point clouds for {num_meshes} meshes")

# %%
plot_pointcound(point_cloud_dataset_collected[4])

# %%
path_to_voxel_matrix = pathlib.Path("data/dataset_voxels.npz")
data = None
if path_to_voxel_matrix.exists():
    data = np.load(path_to_voxel_matrix)["data"]

if not path_to_voxel_matrix.exists():
    voxel_data_generator = (retrieve_voxel_data(cloud, 32) for cloud in tqdm(point_cloud_dataset_collected, total=num_meshes))
    voxel_data_collected = list(voxel_data_generator)
    voxel_data = np.array([item[0] for item in voxel_data_collected])
    data = voxel_data.reshape(voxel_data.shape + (1, ))
    np.savez_compressed(path_to_voxel_matrix, data=data)
data_shape = data.shape[1:]
f"Loaded voxel data for {num_meshes} meshes"

# %%
cut_point = .1
x_train, x_test = train_test_split(data.astype(np.float), shuffle=True, test_size=.1)
f"Training: {x_train.shape} | Test: {x_test.shape}"

# %%
# https://stackoverflow.com/a/58526969/4162265
# https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example
tf.config.run_functions_eagerly(True)


class EncoderUnit(Layer):
    def __init__(self, num_filters, name="encoder_unit", **kwargs) -> None:
        super(EncoderUnit, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv3D(num_filters, 3, activation='elu', strides=2, padding='same', kernel_regularizer=reg.L1L2(.1, .1))
        self.drop = layers.Dropout(.5)
        self.norm = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.drop(x)
        x = self.norm(x)
        return x


class DecoderUnit(Layer):
    def __init__(self, num_filters, name="encoder_unit", **kwargs) -> None:
        super(DecoderUnit, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv3DTranspose(num_filters, 3, activation="elu", strides=2, padding="same", kernel_regularizer=reg.L1L2(.1, .1))
        self.drop = layers.Dropout(.5)
        self.norm = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.drop(x)
        x = self.norm(x)
        return x


class Encoder(Model):
    def __init__(self, data_shape, output_dim=8, EncoderUnit=EncoderUnit, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.data_shape = data_shape
        self.output_dim = output_dim
        self.elayers = [
            EncoderUnit(16),
            EncoderUnit(32),
            EncoderUnit(48),
            EncoderUnit(64),
        ]
        self.sampler = Sampling(self.output_dim)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.elayers:
            x = layer(x)
        self.bridging_shape = list(x.shape[1:])
        z = self.sampler(x)
        return z, self.bridging_shape


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, output_dim, name="Sampler", **kwargs):
        super(Sampling, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.flatten = layers.Flatten()
        self.sample = layers.Dense(self.output_dim, kernel_regularizer=reg.L1L2(.1, .1))

    def call(self, inputs):
        x = self.flatten(inputs)
        z = self.sample(x)
        return z


class Decoder(Model):
    def __init__(self, original_shape, bridging_shape, DecoderUnit=DecoderUnit, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.original_shape = original_shape
        self.bridging_shape = bridging_shape
        self.receiver = layers.Dense(np.prod(bridging_shape), kernel_regularizer=reg.L1L2(.1, .1))
        self.unflattener = layers.Reshape(bridging_shape)
        self.dlayers = [
            DecoderUnit(64),
            DecoderUnit(48),
            DecoderUnit(32),
            DecoderUnit(16),
        ]
        self.final_decoding = layers.Conv3DTranspose(1, 3, activation='sigmoid', strides=1, padding="same")

        # self.run_layers = functools.reduce(lambda x, y: y(x), self.layers)

    def call(self, inputs, **kwargs):
        x = self.receiver(inputs)
        x = self.unflattener(x)
        for layer in self.dlayers:
            x = layer(x)
        decoder_outputs = self.final_decoding(x)
        return decoder_outputs


class AutoEncoder(Model):
    def __init__(self, data_shape, output_dim=300, verbose=False, name="autoencoder", **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.original_shape = data_shape
        self.in_shape = (None, ) + self.original_shape
        self.output_dim = output_dim
        self.encoder = Encoder(self.original_shape, self.output_dim)
        self.encoder.build(self.in_shape)
        self.decoder = Decoder(self.original_shape, self.encoder.bridging_shape)


    def call(self, inputs, **kwargs):
        x = layers.InputLayer(self.original_shape)(inputs)
        z, bridging_shape = self.encoder(x)
        x = layers.InputLayer(self.original_shape)(z)
        decodings = self.decoder(x, bridging_shape=bridging_shape)
        return decodings

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.build((None, ) + self.original_shape)
        self.encoder.summary(line_length=line_length, positions=positions, print_fn=print_fn)
        self.decoder.summary(line_length=line_length, positions=positions, print_fn=print_fn)
        return super().summary(line_length=line_length, positions=positions, print_fn=print_fn)


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    # https://stackoverflow.com/questions/61799546/how-to-custom-losses-by-subclass-tf-keras-losses-loss-class-in-tensorflow2-x
    """
    Args:
    pos_weight: Scalar to affect the positive labels of the loss function.
    weight: Scalar to affect the entirety of the loss function.
    from_logits: Whether to compute loss from logits or the probability.
    reduction: Type of tf.keras.losses.Reduction to apply to loss.
    name: Name of the loss function.
    """
    def __init__(self, penalty, reduction=tf.keras.losses.Reduction.AUTO, name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.penalty = penalty

    def call(self, y_true, y_pred):
        return WeightedBinaryCrossEntropy.wbce(y_true, y_pred, self.penalty)[0]

    @staticmethod
    def wbce(y_true, y_pred, penalty):
        penalty *= 100
        clipped_y_pred = WeightedBinaryCrossEntropy.clip_pred(y_pred, 1e-7, 1.0 - 1e-7)
        binary_cross_entropy = -(penalty * y_true * tf.math.log(clipped_y_pred) + (100 - penalty) * (1.0 - y_true) * tf.math.log(1.0 - clipped_y_pred)) / 100.0
        loss = tf.reduce_sum(binary_cross_entropy)
        return loss, binary_cross_entropy

    @staticmethod
    def clip_pred(y_pred, min_val, max_val):
        clipped_y_pred = tf.keras.backend.clip(y_pred, min_val, max_val)
        return clipped_y_pred


def _rebase(x1, x2, min_val, max_val):
    x1 = (x1 * (max_val - min_val)) + min_val
    x2 = (x2 * (max_val - min_val)) + min_val
    return x1, x2


# if verbose > 0:
penalty = .90
learning_rate = 0.01
comment = "refactored run"
lbound, ubound = -1, 2
batch_size = 5
output_dim = 300

autoencoder = AutoEncoder(data_shape=data_shape, output_dim=output_dim, verbose=2)
loss_fn = WeightedBinaryCrossEntropy(penalty=penalty)
opt_cl = opt.SGD(learning_rate=learning_rate, nesterov=True, momentum=.9)
opt_cl = opt.Adam(learning_rate=learning_rate)

autoencoder.compile(optimizer=opt_cl, loss=loss_fn)
autoencoder.summary()
# %%
x_train_mod, y_train_mod = _rebase(x_train, x_train, 0, 1)
x_test_mod, y_test_mod = _rebase(x_test, x_test, 0, 1)
x_train_mod, y_train_mod = _rebase(x_train, x_train, lbound, ubound)
x_test_mod, y_test_mod = _rebase(x_test, x_test, 0, 1)
# x_train_mod, _ = _rebase(x_train, x_test, lbound, ubound)
# y_train_mod, _ = _rebase(x_train, x_test, 0, 1)


def construct_log_dir(elements=[]):
    log_dir = f"logs/{datetime.now().strftime('%m%d%H%M%S')}/" + " - ".join(elements)
    return log_dir


elems = [
    f"[{lbound:07.7f} & {ubound:07.7f}]",
    f"[{penalty:05.5f}]",
    f"[{learning_rate:07.7f}]",
    f"[{batch_size:02d}]",
    f"{comment}" if comment else "NA",
]

log_dir = construct_log_dir(elems)
print(log_dir)


def lr_schedule(epoch):
    """
  Returns a custom learning rate that decreases as epochs progress.
  """
    learning_rate = 0.0001
    if epoch > 5:
        learning_rate = 0.001

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def my_ceil(a, precision=0):
    # https://stackoverflow.com/a/58065394/4162265
    return np.round(a + 0.5 * 10**(-precision), precision)


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, penalty=.99, n=5):
        self.x_test, self.y_test = validation_data
        self.n = n
        self.penalty = penalty

    def on_epoch_begin(self, epoch, logs={}):
        # https://stackoverflow.com/questions/51837387/callback-returning-train-and-validation-performance

        test_sample = np.array(random.sample(list(self.x_test), 100))
        val_true = test_sample
        val_predict = np.asarray(self.model.predict(test_sample))
        loss, bce = WeightedBinaryCrossEntropy.wbce(val_true, val_predict, self.penalty)
        min_pred = np.min(val_predict)
        max_pred = np.max(val_predict)
        mean_pred = np.mean(val_predict)
        median_pred = np.median(val_predict)
        minmax_ratio = min_pred / max_pred
        print(f"Min {min_pred:.3f} - Max {max_pred:.3f} - Mean {mean_pred:.3f} - Loss {loss:.3f}")
        # fig = generate_img_of_decodings_expanded(val_true, val_predict, [.20, .35, .50, .80, .9, .95, .99, .999])

        # precision = 3
        # floored_max = my_floor(max_pred, precision)
        # thresh = sorted(floored_max + np.linspace(0, 1, 11) / 10**precision)
        thresh = np.linspace(0.8, 1, 11)
        fig = generate_img_of_decodings_expanded(val_true, val_predict, thresh)
        fig.tight_layout()
        fig.savefig('example.png')  # save the figure to file
        plt.close()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf, np.uint8)

        tf.summary.histogram("predictions", val_predict, step=epoch)
        tf.summary.scalar("min", min_pred, step=epoch)
        tf.summary.scalar("max", max_pred, step=epoch)
        tf.summary.scalar("mean", mean_pred, step=epoch)
        tf.summary.scalar("median", median_pred, step=epoch)
        tf.summary.scalar("minmax", minmax_ratio, step=epoch)
        tf.summary.histogram("BCE", bce, step=epoch)
        tf.summary.scalar("Loss", loss, step=epoch)
        tf.summary.image(f"After epoch: {epoch}", img.reshape(-1, *img.shape), step=epoch)
        tf.summary.flush()


file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
validation_data = (x_test_mod, y_test_mod)
image_log_callback = CustomCallback(validation_data, penalty=penalty)
history = autoencoder.fit(
    x_train_mod,
    y_train_mod,
    epochs=100,
    batch_size=batch_size,
    shuffle=True,
    validation_data=validation_data,
    callbacks=[
        image_log_callback,
        # lr_callback,
        # tensorboard_callback,
    ])
# %%
encoder.save("models/3d_encoder.h5")
decoder.save("models/3d_decoder.h5")
autoencoder.save("models/3d_autoencoder.h5")
# %%
mpath = pathlib.Path("models/3d_encoder.h5")
if mpath.exists():
    encoder = tf.keras.models.load_model(mpath)

mpath = pathlib.Path("models/3d_decoder.h5")
if mpath.exists():
    decoder = tf.keras.models.load_model(mpath)

mpath = pathlib.Path("models/3d_autoencoder.h5")
if mpath.exists():
    autoencoder = tf.keras.models.load_model(mpath, compile=False, custom_objects={'WeightedBinaryCrossEntropy': WeightedBinaryCrossEntropy(penalty), 'penalty': penalty})

# %%

n = 5
test_sample = np.array(random.sample(list(x_train), n))
z = encoder.predict(test_sample)

originals = test_sample.reshape(test_sample.shape[:-1])
decodings = decoder.predict(z).reshape(test_sample.shape[:-1])
# %%
generate_img_of_decodings(originals, decodings, thresh=.7)
plt.show()
