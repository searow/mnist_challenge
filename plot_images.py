import numpy as np
from matplotlib import pyplot
import pdb
from random import randint

attack_file = "attacks/wrong_examples.npy"

#"attacks/20181211-160454-0.04-1.0-0.3-0.01-128-secret-clipped_pixels-128-attack.npy"

x = np.load(attack_file)
nx = x.size/784

pdb.set_trace()
x = np.squeeze(x)
x = np.reshape(x, (nx,28,28))

rand = randint(0,nx)
while(True):
    #pdb.set_trace()
    pyplot.imshow(x[rand], cmap='gray')
    print(rand)
    pyplot.show()
    rand = randint(0,nx)
