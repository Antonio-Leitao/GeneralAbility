import os
from tensorflow.keras import models
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Keras não incomodar com logs


def keras_model(shape, loss, metrics, X_shape):
    model = models.Sequential()
    model.add(layers.Dense(shape[0][0], activation=shape[0][1], input_shape=X_shape))

    for layer in shape[1:]:
        model.add(layers.Dense(layer[0], activation=layer[1]))

    model.compile(optimizer='sgd', loss=loss, metrics=metrics)

    return model
