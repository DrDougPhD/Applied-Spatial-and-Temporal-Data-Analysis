import csv
import pprint
import matplotlib.pyplot as plt
import numpy as np


# load data
lines = []
with open('results.csv') as f:
    reader = csv.DictReader(f, delimiter=';')
    lines = [line for line in reader]

# transform to be friendly for plotting
methods = [r['method'] for r in lines]
mae_fold1 = [r['mae-fold1'] for r in lines]
mae_fold2 = [r['mae-fold2'] for r in lines]
mae_fold3 = [r['mae-fold3'] for r in lines]
mae_avg   = [r['mae-avg'] for r in lines]
mae_vals = [mae_fold1, mae_fold2, mae_fold3, mae_avg]

rmse_fold1 = [r['rmse-fold1'] for r in lines]
rmse_fold2 = [r['rmse-fold2'] for r in lines]
rmse_fold3 = [r['rmse-fold3'] for r in lines]
rmse_avg = [r['rmse-avg'] for r in lines]
rmse_vals = [rmse_fold1, rmse_fold2, rmse_fold3, rmse_avg]

labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Mean']
num_bars = len(mae_fold1)

# create two subplot figure
fig, (mae_axes, rmse_axes) = plt.subplots(2, sharex=True)
mae_axes.set_ylabel('MAE')
rmse_axes.set_ylabel('RMSE')


# draw subplot for mae
bar_slots_to_occupy = num_bars + 1
for bar_offset, result_group in enumerate(mae_vals):
    # draw bars for fold k / average results
    indices = np.arange(num_bars)+1+(bar_offset*bar_slots_to_occupy)


# draw subplot for rmse
for result_group in rmse_vals:
    pass
    # draw bars for fold k / average results


# apply legend


plt.show()
