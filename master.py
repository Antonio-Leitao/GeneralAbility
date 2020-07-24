import time
import os
import numpy as np
from loss_functions import gen_loss, density_prob, density_loss
from callbacks import gen_score
from models import keras_model
from benchmark_class import keras_benchmark

from sklearn.datasets import make_regression


model_shape = [[4, 'relu'] * 4, [1, 'linear']]
metrics = ['mae']


X, y = make_regression(1000, 20)
data = []#[X, y]
partition_ratio = .33

epochs = 100
seeds = range(10)
partition_seed = seeds
datasets = ['CONCRETE']

print('custom')
filename = 'make_regression_density_custom'
benchmark_k = keras_benchmark(keras_model, model_shape, density_loss, gen_score, metrics, partition_ratio)

tik = time.time()
t = benchmark_k.benchmark(seeds, epochs, datasets, filename, partition_seed, example=data)
print(time.time() - tik)

print('control')
filename = 'make_regression_density_control'

benchmark_k = keras_benchmark(keras_model, model_shape, 'mae', gen_score, metrics, partition_ratio)

tik = time.time()
t = benchmark_k.benchmark(seeds, epochs, datasets, filename, partition_seed, example=data)
print(time.time() - tik)