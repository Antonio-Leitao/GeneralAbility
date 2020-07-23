import time
import os
import numpy as np
from loss_functions import gen_loss, density_prob, density_loss
from callbacks import gen_score
from models import keras_model
from benchmark_class import keras_benchmark

from sklearn.datasets import make_regression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Keras não incomodar com logs

partition_ratio = .33
partition_seed = 42

model_shape = [[10, 'relu'] * 5, [1, 'linear']]
metrics = ['mae']


X, y = make_regression(2000, 20)
data = [X, y]

epochs = 200
seeds = range(100)
datasets = []

print('custom')
filename = 'make_regression_custom'
benchmark_k = keras_benchmark(keras_model, model_shape, gen_loss, gen_score, metrics, partition_ratio, partition_seed)

tik = time.time()
t = benchmark_k.benchmark(seeds, epochs, datasets, filename, example=data)
print(time.time() - tik)

print('control')
filename = 'make_regression_control'

benchmark_k = keras_benchmark(keras_model, model_shape, 'mae', gen_score, metrics, partition_ratio, partition_seed)

tik = time.time()
t = benchmark_k.benchmark(seeds, epochs, datasets, filename, example=data)
print(time.time() - tik)