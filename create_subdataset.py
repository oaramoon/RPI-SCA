import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import pickle


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)



indices_8 = np.where(train_labels == 8)[0]
training_8s = train_images[indices_8,::]
print(len(indices_8))
indices_1 = np.where(train_labels == 1)[0]
training_1s = train_images[indices_1,::]
print(len(indices_1))

with open("1_vs_8.pkl",'wb') as f:
    pickle.dump({"8":training_8s[0:1000,::], "1":training_1s[0:1000,::]},f)


