import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

save_dir = 'original_images'
orig_images_filename = os.path.join(save_dir, 'orig_images.npy')
orig_labels_filename = os.path.join(save_dir, 'orig_labels.npy')

# Load MNIST data.
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
orig_images = mnist.test.images
orig_labels = mnist.test.labels

os.makedirs(save_dir, exist_ok=True)
np.save(orig_images_filename, orig_images)
np.save(orig_labels_filename, orig_labels)