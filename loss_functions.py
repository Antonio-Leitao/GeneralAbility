import tensorflow as tf
import tensorflow.keras.backend as K


def gen_loss(distance_matrix):
    def loss(y_true, y_pred):
        distances = K.constant(distance_matrix[:, :-1], name='distance_matrix')

        errors_difference = K.abs(K.transpose(K.abs(y_true - y_pred)) - K.abs(y_true - y_pred))

        errors_by_distance = tf.math.divide(errors_difference, distances + K.constant(1), name='division')

        p_x = K.mean(K.exp(-errors_by_distance), axis=1)

        score = K.abs(y_true - y_pred) * p_x

        return K.mean(score)

    return loss


def density_prob(distance_matrix):
    distances = K.constant(distance_matrix, name='distance_matrix')

    density_score = K.mean(K.exp(-distances), axis=1)

    return density_score


def density_loss(density_score):
    def loss(y_true, y_pred):

        score = K.abs(y_true - y_pred) * density_score

        return K.mean(score)

    return loss
