import json
import pprint
import matplotlib.pyplot as plt
import numpy as np


# load data
lines = []
with open('results.json') as f:
    results = json.load(f)


# transform to be friendly for plotting
mae_results = {
    method: results[method]['mae']
    for method in results
}
rmse_results = {
    method: results[method]['rmse']
    for method in results
}
access_order = list(results.keys())
labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Mean']

num_bars_per_method = len(labels)
num_bars_per_group = 5


# create two subplot figure
fig, (mae_axes, rmse_axes) = plt.subplots(2, sharex=True)
mae_axes.set_ylabel('MAE')
rmse_axes.set_ylabel('RMSE')


# draw subplot for mae
bar_slots_to_occupy = num_bars_per_group + 1
base_indices = np.arange(start=1,
                         stop=bar_slots_to_occupy*len(labels)+1,
                         step=bar_slots_to_occupy)
for i, method in enumerate(access_order):
    # draw bars for fold k / average results
    indices = base_indices + i
    print(indices)


# apply legend


plt.show()
