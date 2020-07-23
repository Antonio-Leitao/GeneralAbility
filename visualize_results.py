import numpy as np
import pandas as pd


path = 'results/'

custom_filename = 'make_regression_custom'
control_filename = 'make_regression_control'


def results(array):
    df = pd.DataFrame([])
    for _ in range(len(array)):
        seed = int(array[_][0])
        dataset = array[_][1]
        train_mae = np.mean(np.abs(array[_][2]))
        test_mae = np.mean(np.abs(array[_][3]))
        gen_score = np.mean(np.abs(array[_][3]) * np.abs(array[_][4]))
        df = df.append([[seed, dataset, train_mae, test_mae, gen_score]])

    df.columns = ['seed', 'dataset', 'Train Mae', 'Test Mae', 'Gen Score']
    return df


custom_array = np.load(f'results/{custom_filename}.npy', allow_pickle='True')
control_array = np.load(f'results/{control_filename}.npy', allow_pickle='True')

custom_result = results(custom_array)
control_result = results(control_array)

print(f'\nAverage over {len(custom_result)} seeds:')
print('Custom:\t\t', 'Train Mae: ', custom_result['Train Mae'].mean().round(4), 'Test Mae: ', custom_result['Test Mae'].mean().round(4), '\tGen Score: ', custom_result['Gen Score'].mean().round(4))
print('Control:\t', 'Train Mae: ', control_result['Train Mae'].mean().round(4), 'Test Mae: ', control_result['Test Mae'].mean().round(4), '\tGen Score: ', control_result['Gen Score'].mean().round(4))

