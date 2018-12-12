import numpy as np
from matplotlib import pyplot
import pdb
from random import randint

orig_images_file = "original_images/orig_images.npy"
orig_labels_file = "original_images/orig_labels.npy"
attack_images_file = "attacks/20181212-014840-0.01-1.0-0.3-0.01-1-adv_trained-top_k_distrib_grads-5-attack.npy"
attack_labels_file = "attacks/20181212-014840-0.01-1.0-0.3-0.01-1-adv_trained-top_k_distrib_grads-5-ypred.npy-pred.npy"
true_labels_file = ""

orig_images = np.load(orig_images_file)
orig_labels = np.load(orig_labels_file)
attack_images = np.load(attack_images_file)
attack_labels = np.load(attack_labels_file)
# true_labels = np.load(true_labels_file)
true_labels = np.zeros_like(orig_labels)

orig_images = np.reshape(orig_images, (-1, 28, 28))
attack_images = np.reshape(attack_images, (-1, 28, 28))

assert orig_images.shape[0] == attack_images.shape[0]

for i in range(orig_images.shape[0]):
    fig = pyplot.figure()
    ax_orig = fig.add_subplot(1, 2, 1)
    ax_attack = fig.add_subplot(1, 2, 2)

    ax_orig.imshow(orig_images[i, :, :], cmap='gray')
    ax_attack.imshow(attack_images[i, :, :], cmap='gray')

    true_label = true_labels[i]
    orig_label = orig_labels[i]
    attack_label = attack_labels[i]
    title = 'True: {}, Orig: {}, Attack: {}'.format(true_label, orig_label, attack_label)

    fig.suptitle(title)

    ax_orig.axis('off')
    ax_attack.axis('off')

    pyplot.show()
