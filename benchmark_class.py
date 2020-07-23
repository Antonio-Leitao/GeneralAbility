import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class benchmark:
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

        self.batch_size = int(len(self.X_train)/4)

        self.X_test = StandardScaler().fit_transform(self.X_test)
        self.y_test = StandardScaler().fit_transform(self.y_test.reshape(-1, 1))

        self.d_matrix = np.c_[distance_matrix(self.X_train, self.X_train), self.y_train]

        if not isinstance(self.loss, str):
            built_loss = self.loss(self.d_matrix, self.batch_size)
        else:
            built_loss = self.loss


        self.model = self.model_function(self.model_shape, built_loss, self.metric,
                                         (self.X_train.shape[1],))

        self.callback = self.callback(train=(self.X_train, self.y_train), test=(self.X_test, self.y_test),
                                      d_matrix=self.d_matrix)

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
                                             epochs=epochs, batch_size=self.batch_size, verbose=1, callbacks=[self.callback])
                    self.results.append([seed, history.history])

                    return self.results


test = GEN_NN_benchmark(model_create, [[10, 'relu'] * 5, [1, 'linear']], custom_loss_1, ['mae'], Generalization)

import time

tik = time.time()
t = test.benchmark([1], 20, ['Concrete'])
print(time.time() - tik)