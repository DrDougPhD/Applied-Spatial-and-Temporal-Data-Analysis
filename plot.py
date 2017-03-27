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
fig, (dummy_axes, mae_axes, rmse_axes) = plt.subplots(3, sharex=True)
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

    rmse_values = list(rmse_results[method])
    rmse_values.append(np.mean(rmse_values))
    rmse_axes.bar(indices, rmse_values, label=method)

    mae_values = list(mae_results[method])
    mae_values.append(np.mean(mae_values))
    mae_axes.bar(indices, mae_values, label=method)


# apply legend
"""
handles, labels = mae_axes.get_legend_handles_labels()
mae_axes.legend(handles, labels,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0)
"""
plt.sca(mae_axes)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           mode='expand', borderaxespad=1.)
plt.show()
