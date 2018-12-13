import numpy as np
from matplotlib import pyplot
import pdb
import json
from random import randint

pgd_attack_file = "attacks/attacks/20181212-012544-0.01-1.0-0.3-0.01-100-adv_trained-pgd-51-128-summary.txt"
clipped_attack_file = "attacks/attacks/20181212-012544-0.01-1.0-0.3-0.01-100-adv_trained-pgd-51-128-summary.txt"

with open(pgd_attack_file) as f:
    pgd_attack = json.load(f)
with open(clipped_attack_file) as f:
    clp_attack = json.load(f)

pgd_acc = pgd_attack["accuracy_iters"]
clp_acc = clp_attack["accuracy_iters"]
iterations = len(pgd_acc)


fig = pyplot.figure()
pyplot.plot(list(range(iterations)), pgd_acc, clp_acc)

title = 'PGD and Clipped Pixels Accuracies vs Random Restart Iteration'

fig.suptitle(title)

# Show plot
#pyplot.show()

# Save plot
fig.savefig("PGD_CLIP_trainingplot")
pyplot.close()
