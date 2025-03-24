import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--gpuid", type=str, default=0)
parser.add_argument("--loadmodel", action="store_true")
parser.add_argument("--removeout", action="store_true")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--bs", type=int, default=512)
parser.add_argument("--modeln", type=str, default="name")
parser.add_argument("--loss", type=str, default="mse")

args = parser.parse_args()

# gpuid = args.gpuid
remove = args.removeout
loadmodel = args.loadmodel
lr = args.lr
bs = args.bs
modeln = args.modeln
loss = args.loss

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

os.environ["OMP_NUM_THREADS"] = "6"
#del os.environ["OMP_NUM_THREADS"]

import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
print('numpy: %s' % np.__version__)

# Tensorflow/Keras
import tensorflow as tf
from tensorflow import keras
print('Keras: %s' % keras.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import Input, layers
from keras import backend as K

# Sklearn
import sklearn
print('sklearn: %s' % sklearn.__version__)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle

N = 12        # Maximum number of atoms in unit cell

ns = 1344     # Size of dc descriptors

strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    inp = keras.Input(shape=(ns + 3,))

    x = layers.Dense(1024)(inp)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Dense(1024)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.Dropout(0.5)(x)

    x = layers.Dense(1024)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.Dropout(0.5)(x)

    outp = layers.Dense(2)(x)

    smodel = keras.Model(inp, outp, name="potential")

    inps = [keras.Input(shape=(ns + 3,)) for u in range(N)]
    mr = [keras.Input(shape=(1,)) for u in range(N)]
    mi = [keras.Input(shape=(1,)) for u in range(N)]

    vv = [smodel(u) for u in inps]

    vr = [y[:, 0] for y in vv]
    vi = [y[:, 1] for y in vv]

    rs1 = [layers.Multiply()([u, v]) for (u, v) in zip(mr, vr)]
    rs2 = [layers.Multiply()([u, v]) for (u, v) in zip(mi, vi)]

    is1 = [layers.Multiply()([u, v]) for (u, v) in zip(mr, vi)]
    is2 = [layers.Multiply()([u, v]) for (u, v) in zip(mi, vr)]

    Svr = [layers.Subtract()([u, v]) for (u, v) in zip(rs1, rs2)]
    Svi = [layers.Add()([u, v]) for (u, v) in zip(is1, is2)]

    outputs = [layers.Add()(Svr), layers.Add()(Svi)]

    model = keras.Model(inputs=inps + mr + mi, outputs=outputs)

    model.compile(loss=loss, optimizer=Adam(lr))

if loadmodel:
    try:
        with strategy.scope():
            model = tf.keras.models.load_model(f'{modeln}.save')
            # model.compile(loss=loss, optimizer=Adam(lr))
            # model = tf.keras.models.load_weight(f'{modeln:03d}.save')
    except:
        pass

print(model.summary())

dc = np.load("./dc_all.npy")
data = np.load("./d_s.npy")

lent = int(len(data) * 0.8)

data_t = data[:lent]
data_v = data[lent:]


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data):
        self.data = data
        self.indarr = np.arange(len(self.data))

    def __len__(self):
        return int(np.floor(len(self.data) / bs))

    def on_epoch_end(self):
        np.random.shuffle(self.indarr)  # shuffle inplace

    def __getitem__(self, index):
        ind = self.indarr[index * bs : (index+1) * bs]
        td = self.data[ind, :]

        id = td[:, :N].astype(int)
        p = dc[id, :]
        g = td[:, N: N + 3]
        mr = 1.e3 * td[:, N + 3 + 0:-2 - 1:2]
        mi = 1.e3 * td[:, N + 3 + 1:-2 - 0:2]
        Vr = 1.e3 * td[:, -2]
        Vi = 1.e3 * td[:, -1]

        X = ([np.hstack((p[:, i, :], g)) for i in range(N)] +
             [mr[:, i] for i in range(N)] +
             [mi[:, i] for i in range(N)])

        Y = [Vr, Vi]

        return X, Y


train_gen = DataGenerator(data_t)
val_gen = DataGenerator(data_v)

checkpoint_callback = ModelCheckpoint(
    filepath=f'{modeln}.save',
    monitor='val_loss',
    # monitor='loss',
    mode='min',
    save_freq='epoch',
    save_weight_only=False,
    verbose=1,
    save_best_only=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.90,
    patience=20,
    cooldown=0,
    verbose=1,
    min_lr=1e-6,
)

for i in range(3):
    if i == 0:
        lr = 3e-4
        pat = 30
    if i == 1:
        lr = 3e-5
        pat = 60
    if i == 2:
        lr = 3e-6
#        pat = 100000
        pat = 100

    with strategy.scope():
        model = model
        model.compile(loss=loss, optimizer=Adam(lr))

    # K.set_value(model.optimizer.lr, lr)

    es = EarlyStopping(
        monitor="val_loss",
        patience=pat,
        verbose=1,
    )

    model.fit_generator(
        generator=train_gen,
        epochs=100000,
        verbose=2,
        validation_data=val_gen,
        max_queue_size=50,
        use_multiprocessing=False,
        workers=6,
        callbacks=[checkpoint_callback, es],
        # callbacks=[checkpoint_callback],
    )
