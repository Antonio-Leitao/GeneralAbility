import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K

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


class Generalization(tf.keras.callbacks.Callback):

    def __init__(self, train, test, d_matrix):
        super(Generalization, self).__init__()
        self.test = test
        self.train = train
        self.dist = d_matrix

    def on_epoch_end(self, epoch, logs={}):
        logs['gen_score'] = float('-inf')
        logs['n_diff'] = float('-inf')
        X_train, y_train = self.train[0], self.train[1]
        X_test, y_test = self.test[0], self.test[1]

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        k = len(y_train_pred) - 1
        # C = np.min(self.dist, axis=0)
        C = np.array([np.mean(self.dist[:, i][np.argsort(self.dist[:, i], axis=0)][:k], axis=0) for i in
                      range(self.dist.shape[1])])
        NN = np.array([np.argsort(self.dist[:, i], axis=0)[:k] for i in range(self.dist.shape[1])])

        e = [np.mean([np.exp(
            -np.divide(np.abs(np.abs(y_test_pred[i] - y_test[i]) - np.abs(y_train_pred[j] - y_train[j])),
                       self.dist[j, i] + 1)) for j in NN[i]]) for i in range(len(y_test))]
        M = np.abs(y_test_pred - y_test) * e
        score = np.mean(M)
        logs['gen_score'] = np.round(score, 5)


class GEN_NN_benchmark_:
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

        self.d_matrix = distance_matrix(self.X_train, self.X_test)

        self.model = self.model_function(self.model_shape, self.loss, self.metric, (self.X_train.shape[1],))

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
                    history = self.model.fit(self.X_train, self.y_train, validation_data=[self.X_test, self.y_test],
                                             epochs=epochs, batch_size=16, verbose=0, callbacks=[self.callback])
                    self.results.append([seed, history.history])

                    return self.results


test = GEN_NN_benchmark_(model_create, [[1, 'relu'], [1, 'linear']], 'mse', ['mae'], Generalization)

t = test.benchmark([1], 20, ['BIOAVAILABILITY'])

print(t)