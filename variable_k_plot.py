import json
import pprint
import matplotlib.pyplot as plt
import numpy as np
import collections

HATCHES = ('/', '', '*', 'o', 'O', '.')


# load data
lines = []
with open('kvaried.json') as f:
    results = json.load(f)


# transform to be friendly for plotting
indices = None
mae_averages = collections.defaultdict(list)
rmse_averages = collections.defaultdict(list)
for method, values in results.items():
    if indices is None:
        indices = [k for k, _ in values]

    mae_averages[method] = [np.mean(values_for_k['mae'])
                            for k, values_for_k in values]
    rmse_averages[method] = [np.mean(values_for_k['rmse'])
                             for k, values_for_k in values]

pprint.pprint(mae_averages)
pprint.pprint(rmse_averages)

# create two subplot figure
fig, (mae_axes, rmse_axes) = plt.subplots(2, sharex=True)
mae_axes.set_ylabel('MAE (avg)')
rmse_axes.set_ylabel('RMSE (avg)')


# draw subplot
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
majorLocator = MultipleLocator(20)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(5)

optimal_points = set()
for method in mae_averages:
    print('-'*80)
    line, = rmse_axes.plot(indices, rmse_averages[method], label=method+' (using MSD)')
    optimal_rmse = min(zip(indices, rmse_averages[method]),
                       key=lambda x: x[1])
    rmse_axes.axvline(x=optimal_rmse[0], color=line.get_color())
    print("Optimal k for {0} (RMSE): {1}".format(
        method,
        optimal_rmse
    ))
    optimal_points.add(optimal_rmse[0])

    line, = mae_axes.plot(indices, mae_averages[method], label=method+' (using MSD)')
    optimal_mae = min(zip(indices, mae_averages[method]),
                      key=lambda x: x[1])
    optimal_points.add(optimal_mae[0])
    mae_axes.axvline(x=optimal_mae[0], color=line.get_color())
    print("Optimal k for {0} (MAE): {1}".format(
        method,
        optimal_mae
    ))

# apply legend
indices = sorted([*list(range(0, max(indices)+1, 20)),
                  *list(optimal_points)])
indices.remove(40)
print('Indices for RMSE:', indices)
rmse_axes.set_xticks(indices)

plt.suptitle('Problem 14: Impact of Number of Neighbors on Collaborative Filtering')
plt.sca(rmse_axes)
plt.legend(bbox_to_anchor=(0., -.65, 1., .102), loc=8,
           borderaxespad=1.)
#plt.xticks(indices)
plt.subplots_adjust(bottom=0.2)
plt.savefig('sim_vark.svg')

# create a table of averages
"""
with open('sim_vark.txt', 'w') as f:
    f.write('{0: >35} & {1: >10} & {2} & {3} \\\\ \n'
            '\hline \\\\ \n'.format(
        'Method', 'Measurement', 'RMSE (Mean)', 'MAE (Mean)'))

    for method, values_by_metric in results.items():
        for metric, values in values_by_metric.items():
            f.write('{0: >35} &  {1: >11} & {2:.4f}   &   {3:.4f}    \\\\ \n'.format(
                    method, metric,
                    np.mean(values['rmse']),
                    np.mean(values['mae'])))
"""

