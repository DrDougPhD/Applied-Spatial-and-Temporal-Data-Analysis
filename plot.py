import json
import pprint
import matplotlib.pyplot as plt
import numpy as np

HATCHES = ('/', '', '*', 'o', 'O', '.')


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

# define the order for plotting based on descending average
methods = list(results.keys())
avg_scores = [np.mean(mae_results[m]) for m in methods]
itemized_scores = sorted(zip(methods, avg_scores),
                         key=lambda x: x[1],
                         reverse=True)
access_order = [x[0] for x in itemized_scores]


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

    rmse_values = list(rmse_results[method])
    rmse_values.append(np.mean(rmse_values))
    rmse_axes.bar(indices, rmse_values, label=method, hatch=2*HATCHES[i])

    mae_values = list(mae_results[method])
    mae_values.append(np.mean(mae_values))
    mae_axes.bar(indices, mae_values, label=method, hatch=2*HATCHES[i])


# apply legend
"""
handles, labels = mae_axes.get_legend_handles_labels()
mae_axes.legend(handles, labels,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0)
"""

#rmse_axes.set_xticks(base_indices+2)
#rmse_axes.set_xticklabels(labels)

plt.sca(rmse_axes)
plt.legend(bbox_to_anchor=(0., -1.4, 1., .102), loc=8,
           borderaxespad=1.)
plt.xticks(base_indices+2, labels)
plt.subplots_adjust(bottom=0.35)
plt.show()
