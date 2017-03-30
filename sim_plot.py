import json
import pprint
import matplotlib.pyplot as plt
import numpy as np

HATCHES = ('/', '', '*', 'o', 'O', '.')


# load data
lines = []
with open('similarities.json') as f:
    results = json.load(f)


# transform to be friendly for plotting
mae_results = [
    [np.mean(results[method][sim]['mae'])
     for sim in results[method]]
    for method in results
]
rmse_results = [
    [np.mean(results[method][sim]['rmse'])
     for sim in results[method]]
    for method in results
]
access_order = list(results.keys())

sample_method = list(results.keys())[0]
sim_order = list(results[sample_method].keys())


# define the order for plotting based on descending average
num_bars_per_method = 3
num_bars_per_group = 2


# create two subplot figure
fig, (mae_axes, rmse_axes) = plt.subplots(2, sharex=True)
mae_axes.set_ylabel('MAE (avg)')
rmse_axes.set_ylabel('RMSE (avg)')


# draw subplot for mae
bar_slots_to_occupy = num_bars_per_group + 1
base_indices = np.arange(start=1,
                         stop=bar_slots_to_occupy*num_bars_per_method+1,
                         step=bar_slots_to_occupy)
for i, method in enumerate(access_order):
    # draw bars for fold k / average results
    indices = base_indices + i

    rmse_axes.bar(indices, rmse_results[i], label=method, hatch=2*HATCHES[i])

    mae_axes.bar(indices, mae_results[i], label=method, hatch=2*HATCHES[i])


# apply legend
plt.sca(rmse_axes)
plt.legend(bbox_to_anchor=(0., -.65, 1., .102), loc=8,
           borderaxespad=1.)
mae_axes.set_ylim(bottom=0.75)
rmse_axes.set_ylim(bottom=0.95)
plt.xticks(base_indices+0.5, sim_order)
plt.suptitle('Problem 13: Impact of Similarity Measure for Collaborative Filtering')
plt.subplots_adjust(bottom=0.2)
plt.savefig('sim_comp.svg')

# create a table of averages
with open('sim_averages.txt', 'w') as f:
    f.write('{0: >35} & {1: >10} & {2} & {3} \\\\ \n'
            '\hline \\\\ \n'.format(
        'Method', 'Measurement', 'RMSE (Mean)', 'MAE (Mean)'))

    for method, values_by_metric in results.items():
        for metric, values in values_by_metric.items():
            f.write('{0: >35} &  {1: >11} & {2:.4f}   &   {3:.4f}    \\\\ \n'.format(
                    method, metric,
                    np.mean(values['rmse']),
                    np.mean(values['mae'])))

