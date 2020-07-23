import time
import os
from loss_functions import gen_loss, density_prob, density_loss
from callbacks import gen_score
from models import keras_model
from benchmark_class import keras_benchmark

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Keras não incomodar com logs

partition_ratio = .33
partition_seed = 42

model_shape = [[10, 'relu'] * 2, [1, 'linear']]
metrics = ['mae']


epochs = 1
filename = 'CONCRETE'
datasets = ['CONCRETE']
seeds = range(10)


benchmark_k = keras_benchmark(keras_model, model_shape, gen_loss, gen_score, metrics, partition_ratio, partition_seed)

tik = time.time()
t = benchmark_k.benchmark(seeds, epochs, datasets, filename)
print(time.time() - tik)
