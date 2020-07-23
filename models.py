from tensorflow.keras import models
from tensorflow.keras import layers




def keras_model(shape, loss, metrics):
    model = models.Sequential()

    for layer in shape:
        model.add(layers.Dense(layer[0], activation=layer[1]))

    model.compile(optimizer='sgd', loss=loss, metrics=metrics)

    return model
