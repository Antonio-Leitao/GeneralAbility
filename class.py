import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras import models
from tensorflow.keras import layers


def model_create(shape, loss, metrics, X_shape):
    model = models.Sequential()
    model.add(layers.Dense(shape[0][0], activation=shape[0][1], input_shape=X_shape))

    for layer in shape[1:]:
        model.add(layers.Dense(layer[0], activation=layer[1]))

    model.compile(optimizer='sgd', loss=loss, metrics=metrics)

    return model


def custom_loss_1(train, d_matrix):
    def loss(y_true, y_pred):
        # X_train = train[0]

        dist = K.equal(y_true, d_matrix[:, -1])
        b = tf.boolean_mask(d_matrix[:, :-1], dist)

        k = y_pred.shape[0]

        if not k:
            k=1

        distances = K.repeat_elements(K.flatten(b), rep=k, axis=0)

        distances = K.cast(distances, dtype='float32')

        print(y_pred.shape)
        print(K.repeat_elements(K.abs(y_true - y_pred), rep=k, axis=0).shape)
        print(K.concatenate([K.abs(y_true - y_pred)] * k, axis=-1).shape)
        print(distances.shape)
        print('\n')

        estimation = K.exp(-(K.repeat_elements(K.abs(y_true - y_pred), rep=k, axis=0) - K.concatenate(
            [K.abs(y_true - y_pred)] * k, axis=-1)) / (1 + distances))

        score = K.reshape(estimation, shape=(k, k))

        top = K.mean(score, axis=1)

        mul = K.abs(y_true - y_pred) * top

        return K.mean(mul)
    return loss

def custom_loss_2(train, d_matrix):
    def loss(y_true, y_pred):
        X_train = train[0]
        n = K.shape(y_pred)[0:1]
        k = len(X_train)
        print(n, k)

        NN = np.array([np.argsort(d_matrix[:, i], axis=0)[:k] for i in range(d_matrix.shape[1])])

        estimation = (K.repeat_elements(K.abs(y_true-y_pred), rep=k, axis=0)-K.concatenate([K.abs(y_true-y_pred)] * n, axis=-1))

        print(estimation)

        p_x = K.mean(K.exp(-estimation))

        K.mean(K.abs(y_pred - y_true)*p_x)

        e = [K.mean([K.exp(
            -np.divide(K.abs(K.abs(y_pred[i] - y_true[i]) - K.abs(y_pred[j] - y_true[j])),
                       d_matrix[j, i] + 1)) for j in NN[i]]) for i in range(y_true.shape[1])]
        M = K.abs(y_pred - y_true) * e
        score = K.mean(M)

        return score

    return loss


def custom_loss_2(d_matrix):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + K.square(), axis=-1)

    return loss


class Generalization(tf.keras.callbacks.Callback):

    def __init__(self, train, test, d_matrix):
        super(Generalization, self).__init__()
        self.test = test
        self.train = train
        self.dist = d_matrix

    def on_epoch_end(self, epoch, logs={}):
        logs['gen_score'] = float('-inf')

        X_train, y_train = self.train[0], self.train[1]
        X_test, y_test = self.test[0], self.test[1]

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        k = len(y_train_pred) - 1

        NN = np.array([np.argsort(self.dist[:, i], axis=0)[:k] for i in range(self.dist.shape[1])])

        e = [np.mean([np.exp(
            -np.divide(np.abs(np.abs(y_test_pred[i] - y_test[i]) - np.abs(y_train_pred[j] - y_train[j])),
                       self.dist[j, i] + 1)) for j in NN[i]]) for i in range(len(y_test))]
        M = np.abs(y_test_pred - y_test) * e
        score = np.mean(M)
        logs['gen_score'] = np.round(score, 5)


class GEN_NN_benchmark:
    def __init__(self, model_function, model_shape, loss_function, metrics, callback):
        self.model_shape = model_shape
        self.loss = loss_function
        self.metric = metrics
        self.model_function = model_function
        self.results = []
        self.callback = callback

    def build(self, X, y, partition_ratio, partition_seed):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=partition_ratio,
                                                                                random_state=partition_seed)

        self.X_train = StandardScaler().fit_transform(self.X_train)
        self.y_train = StandardScaler().fit_transform(self.y_train.reshape(-1, 1))

        self.X_test = StandardScaler().fit_transform(self.X_test)
        self.y_test = StandardScaler().fit_transform(self.y_test.reshape(-1, 1))

        self.d_matrix = np.c_[distance_matrix(self.X_train, self.X_train), self.y_train]

        self.model = self.model_function(self.model_shape, self.loss(self.X_train, self.d_matrix), self.metric, (self.X_train.shape[1],))

        self.callback = self.callback(train=(self.X_train, self.y_train), test=(self.X_test, self.y_test), d_matrix=self.d_matrix)

    def benchmark(self, seeds, epochs, datasets, example=0):

        if example:
            print('a')

        else:
            for dataset in datasets:
                if dataset == 'RESID_BUILD_SALE_PRICE':
                    data = pd.read_csv('data\\' + dataset + '.txt', header=None, sep='     ', error_bad_lines=False)
                else:
                    data = pd.read_csv('data\\' + dataset + '.txt', header=None, sep='\t', error_bad_lines=False)

                X = data[data.columns[:-1]].values
                y = data[data.columns[-1]].values.reshape(-1, 1)

                for seed in seeds:
                    self.build(X, y, .33, seed)
                    history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                                             epochs=epochs, batch_size=16, verbose=1, callbacks=[self.callback])
                    self.results.append([seed, history.history])

                    return self.results


test = GEN_NN_benchmark(model_create, [[10, 'relu']*5, [1, 'linear']], custom_loss_1, ['mae'], Generalization)

t = test.benchmark([1], 20, ['BIOAVAILABILITY'])
