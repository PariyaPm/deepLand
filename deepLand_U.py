
# Import all modules
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import History
import os
import tensorflow as tf
from keras.utils import multi_gpu_model
import pickle

# import theano

# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

# from tempfile import TemporaryFile

if K.backend( )=='tensorflow':
    K.set_image_dim_ordering('th')

def auc_roc(y_true, y_pred):
    #    y_pred = np.argmax(y_pred,axis=1)

    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true, curve='ROC')

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value



def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc



batch_size = 32
epochs = 200

code_path = '.'
data_path = '.'

os.chdir(code_path)

####Clean_up, pre-process the data and organize it as train and test data sets
data_train = np.load( 'x_patches.npy')
labels_train = np.load( 'y_patches.npy')

# add a dimension to the training data
labels_train = np.expand_dims(labels_train, axis=1)

print(labels_train.shape)

# data_train = np.expand_dims(data_train, axis=4)
# labels_train = np.expand_dims(labels_train, axis=4)
print(data_train.shape)
print(labels_train.shape)

# find the patches that are not in the study area
elev = np.genfromtxt('./elev.txt', delimiter=' ', skip_header = 6)
elev_patches = np.array \
    ([elev[i:i + 32, j:j + 32] for i in range(0, elev.shape[0], int(32)) for j in range(0, elev.shape[1], int(32))])
exc_ind = []
for i in range(elev_patches.shape[0]):
    if(np.sum(elev_patches[i] )==-10238976.0):
        exc_ind.append(i)

elev = None
# generate random indices for the partitioned data

range_ind = np.arange(52455)
np.random.seed(1364)
range_ind = [x for x in range_ind if x not in exc_ind]

np.random.shuffle(range_ind)
x_train= data_train[range_ind[:int(len(range_ind ) *.66)] ,: ,: ,:]
x_test = data_train[range_ind[int(len(range_ind ) *.66):] ,: ,: ,:]
y_train = labels_train[range_ind[0:int(len(range_ind ) *.66)] ,: ,: ,:]
y_test= labels_train[range_ind[int(len(range_ind ) *.66):] ,: ,: ,:]

print(x_train.shape[1:])


model = Sequential()

model.add(Conv2D(64 ,(3 ,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', input_shape = x_train.shape[1:]))
model.add(Conv2D(64 ,(3 ,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(128, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(256, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(512, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))

model.add(Conv2D(1024, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(256, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(256, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(UpSampling2D(size=(2 ,2)))

model.add(Conv2D(128, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(128, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(128, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(UpSampling2D(size=(2 ,2)))

model.add(Conv2D(64, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(64, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(64, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))
model.add(Conv2D(2, (3 ,3), activation = LeakyReLU(alpha=1e-1), padding = 'same', kernel_initializer = 'he_normal'))


model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2D(1, 1 ,activation = LeakyReLU(alpha=1e-1)))

model.add(Activation('sigmoid'))

deepLandU = multi_gpu_model(model, gpus=4)
print(deepLandU.summary())

deepLandU.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy',auc_roc,dice_coef])

print(deepLandU.summary())


import time

start_time = time.time()

# Fit model
cnn = deepLandU.fit(x_train, y_train,
                    batch_size= batch_size, epochs=epochs,
                    validation_data=(x_test ,y_test), shuffle=False)


y_predict = deepLandU.predict(x_test, batch_size= batch_size)


end_time = time.time()

print("the running time for fit predict is", end_time -start_time)
# cnn = deepLandU.fit(x_train, y_train,
#               batch_size= batch_size, epochs=epochs,
#               validation_data=(x_test,y_test), shuffle=True)
y_class = y_predict
y_class[y_class>=0.5] = 1
y_class[y_class <0.5] = 0

y_predict = deepLandU.predict(x_test, batch_size=batch_size)

np.save("y_predictdeepLandU",y_predict)
np.save("y_classdeepLandU",y_class)

# load cnnmodel and do predict and class once again

w = deepLandU.get_weights()
np.save('weightsdeepLandU', w)

histdict=cnn.history

import csv

w = csv.writer(open("deepLandU.csv", "w"))
for key, val in histdict.items():
    w.writerow([key, val])