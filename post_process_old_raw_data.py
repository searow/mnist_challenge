import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix

readdir = 'temp'
is_json_result = True
# How many rows and how many cols for subplots. Should multiply to how many of x axis
# types you need.
subplot_shape = (3, 1)
# fig_shape = (16, 10) # Good for 3x3
fig_shape = (5, 10) # Good for 3x1

cwd = os.getcwd()
attack_dir = os.path.join(cwd, readdir)
files = os.listdir(attack_dir)

files_full_path = [os.path.join(attack_dir, filename) for filename in files]

if is_json_result:
  file_data = []
  for filename in files_full_path:
    with open(filename, 'r') as readfile:
      test_data = json.load(readfile)
      file_data.append(test_data)
else:
  file_data = []
  for filename in files_full_path:
    with open(filename, 'r') as readfile:
      test_data = readfile.read().split('\n')
      output = {}
      for data in test_data:
        line = data.split(':')
        key = line[0]
        value = line[1]
        output[key] = value
      file_data.append(output)

k_values = set()
a_values = set()
top_grads_values = set()

for data in file_data:
  k_values.add(data['k'])
  a_values.add(data['a'])
  top_grads_values.add(data['top_grads'])

accuracies = np.zeros((len(k_values), len(a_values), len(top_grads_values)))

k_idx = list(sorted(k_values, key=lambda x: int(x)))
a_idx = list(sorted(a_values, key=lambda x: float(x)))
top_grads_idx = list(sorted(top_grads_values, key=lambda x: int(x)))

for data in file_data:
  k = k_idx.index(data['k'])
  a = a_idx.index(data['a'])
  top_grads = top_grads_idx.index(data['top_grads'])
  accuracy = data['accuracy']
  accuracies[k, a, top_grads] = accuracy

k_np = np.array(k_idx, dtype=np.int32)
a_np = np.array(a_idx, dtype=np.float)
top_grads_np = np.array(top_grads_idx)

# Plotting.
# Surface plots.
'''
k_grid, a_grid = np.meshgrid(k_np, a_np)

fig = plt.figure()
for i in range(top_grads_np.shape[0]):
  acc = accuracies[:, :, i].T
  ax = fig.add_subplot(3, 3, i + 1, projection='3d')
  ax.set_xlabel('Num steps')
  ax.set_ylabel('Step size')
  ax.set_zlabel('Accuracy')
  ax.set_zlim([np.min(accuracies), np.max(accuracies)])
  ax.plot_surface(k_grid, a_grid, acc, cmap=cm.plasma, linewidth=0)
plt.show()
'''

# Overlaid line plots.
colors = ['b', 'g', 'r', 'c', 'm', 'k']
fig = plt.figure(figsize=fig_shape)
for grad in range(top_grads_np.shape[0]):
  for a in range(a_np.shape[0]):
    acc = accuracies[:, a, grad]
    color = colors[a]

    ax = fig.add_subplot(subplot_shape[0], subplot_shape[1], grad + 1)
    ax.plot(k_np, acc, color)
    ax.set_xlabel('Num steps')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([np.min(accuracies), np.max(accuracies)])
  ax.set_title('Top grads: {}'.format(top_grads_np[grad]))

legend_items = ['Step size = {}'.format(a) for a in a_np]
fig.legend(legend_items, loc='lower right')
fig.subplots_adjust(hspace=0.5)
plt.show()

