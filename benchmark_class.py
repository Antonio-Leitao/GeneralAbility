import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class keras_benchmark:
    def __init__(self, model_function, model_shape, loss, callback, metrics, partition_ratio, partition_seed):
        self.model = model_function
        self.model_shape = model_shape
        self.loss_function = loss
        self.metrics = metrics
        self.results = []
        self.callback = callback
        self.partition_ratio = partition_ratio
        self.partition_seed = partition_seed

    def build(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.partition_ratio,
                                                                                random_state=self.partition_seed)

        self.X_train = StandardScaler().fit_transform(self.X_train)
        self.y_train = StandardScaler().fit_transform(self.y_train.reshape(-1, 1))

        self.batch_size = int(len(self.X_train))

        self.X_test = StandardScaler().fit_transform(self.X_test)
        self.y_test = StandardScaler().fit_transform(self.y_test.reshape(-1, 1))

        self.d_train = np.c_[
            distance_matrix(np.c_[self.X_train, self.y_train], np.c_[self.X_train, self.y_train]), self.y_train]

        self.d_test = distance_matrix(self.y_train, self.y_test)

        if not isinstance(self.loss_function, str):
            built_loss = self.loss_function(self.d_train)
        else:
            built_loss = self.loss_function

        self.call = self.callback(train=(self.X_train, self.y_train), test=(self.X_test, self.y_test),
                                  d_matrix=self.d_test)

        self.compiled_model = self.model(self.model_shape, built_loss, self.metrics)

    def benchmark(self, seeds, epochs, datasets, filename, example=0):

        if example:
            self.X = example[0]
            self.y = example[1].reshape(-1,1)

            for seed in seeds:
                self.build()
                history = self.compiled_model.fit(self.X_train, self.y_train,
                                                  # validation_data=(self.X_test, self.y_test),
                                                  epochs=epochs, batch_size=self.batch_size, verbose=0,
                                                  callbacks=[self.call])

                train_pred = self.compiled_model.predict(self.X_train).flatten()
                test_pred = self.compiled_model.predict(self.X_test).flatten()
                test_p_x = history.history['p_score'][-1]

                self.results.append([seed, filename, train_pred, test_pred, test_p_x])

                np.save('results/' + filename, self.results)

            return self.results

        for dataset in datasets:
            if dataset == 'RESID_BUILD_SALE_PRICE':
                data = pd.read_csv('data\\' + dataset + '.txt', header=None, sep='     ', error_bad_lines=False)
            else:
                data = pd.read_csv('data\\' + dataset + '.txt', header=None, sep='\t', error_bad_lines=False)

            self.X = data[data.columns[:-1]].values
            self.y = data[data.columns[-1]].values.reshape(-1, 1)

            for seed in seeds:
                self.partition_seed = seed
                self.build()
                history = self.compiled_model.fit(self.X_train, self.y_train,
                                                  # validation_data=(self.X_test, self.y_test),
                                                  epochs=epochs, batch_size=self.batch_size, verbose=0,
                                                  callbacks=[self.call])

                train_pred = self.compiled_model.predict(self.X_train).flatten()
                test_pred = self.compiled_model.predict(self.X_test).flatten()
                test_p_x = history.history['p_score'][-1]

                self.results.append([seed, dataset, train_pred, test_pred, test_p_x])

                np.save('results/' + filename, self.results)

            return self.results
