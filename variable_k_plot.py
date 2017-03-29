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
for method in mae_averages:
    print("Optimal k for {0} (RMSE): {1}".format(
	method,
        min(zip(indices, rmse_averages[method]), key=lambda x: x[1])
    ))
    print("Optimal k for {0} (MAE): {1}".format(
        method,
        min(zip(indices, mae_averages[method]), key=lambda x: x[1])
    ))
    print('-'*80)
    rmse_axes.plot(indices, rmse_averages[method], label=method)
    mae_axes.plot(indices, mae_averages[method], label=method)


# apply legend
plt.sca(rmse_axes)
plt.legend(bbox_to_anchor=(0., -1.05, 1., .102), loc=8,
           borderaxespad=1.)
#plt.xticks(indices)
plt.subplots_adjust(bottom=0.35)
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

