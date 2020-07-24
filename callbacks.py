import tensorflow as tf
import numpy as np


class gen_score(tf.keras.callbacks.Callback):
    def __init__(self, train, test, d_matrix):
        super(gen_score, self).__init__()
        self.test = test
        self.train = train
        self.dist = d_matrix

    def on_epoch_end(self, epoch, logs={}):
        logs['gen_score'] = float('-inf')
        logs['p_score'] = float(1)

        X_train, y_train = self.train[0], self.train[1]
        X_test, y_test = self.test[0], self.test[1]

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        k = len(y_train_pred)

        nn = np.array([np.argsort(self.dist[:, i], axis=0)[:k] for i in range(self.dist.shape[1])])

        p_x = [np.mean([np.exp(
            -np.divide(np.abs(np.abs(y_test_pred[i] - y_test[i]) - np.abs(y_train_pred[j] - y_train[j])),
                       self.dist[j, i] + 1)) for j in nn[i]]) for i in range(len(y_test))]

        adjusted = np.abs(y_test_pred - y_test) * p_x
        score = np.mean(adjusted)

        logs['gen_score'] = np.round(score, 5)
        logs['p_score'] = p_x
        print(logs['gen_score'], np.mean(p_x), logs['mae'])

