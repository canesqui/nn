from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import pyplot as plt

import tensorflow as tf
import random
import numpy as np


random.seed(619) # use a standard seed to make repeatable

def gen_image(arr):
    two_d = (np.reshape(arr, (28,28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

def make_rmnist(data, n=10):
    """Make a subset of MNIST using n training examples of each digit and
    save into data/rmnist_n.pkl.gz, together with the complete
    validation and test sets.
    """ 

    indices = range(50000)
    random.shuffle(indices)

    values = [(j, data[1][j]) for j in indices]
    indices_subset = [[v[0] for v in values if v[1] == j][:n]
                      for j in range(10)]
    flattened_indices = [i for sub in indices_subset for i in sub]
    random.shuffle(flattened_indices)
    td0_prime = [data[0][j] for j in flattened_indices]
    td1_prime = [data[1][j] for j in flattened_indices]
    td_prime = (td0_prime, td1_prime)
    return (data(td_prime))
    #x_train = x_train(td_prime)
    #y_train = y_train(td_prime)


mnist = tf.keras.datasets.mnist
print('Dataset')
print(mnist)
print('')


(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train_0 = np.where(y_train == 0)[0][0:39]
y_train_1 = np.where(y_train == 1)[0][0:39]
y_train_2 = np.where(y_train == 2)[0][0:39]
y_train_3 = np.where(y_train == 3)[0][0:39]
y_train_4 = np.where(y_train == 4)[0][0:39]
y_train_5 = np.where(y_train == 5)[0][0:39]
y_train_6 = np.where(y_train == 6)[0][0:39]
y_train_7 = np.where(y_train == 7)[0][0:39]
y_train_8 = np.where(y_train == 8)[0][0:39]
y_train_9 = np.where(y_train == 9)[0][0:39]

args = (y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7,
y_train_8, y_train_9)

index = np.concatenate(args)

x_train = x_train[index]
y_train = y_train[index]

print(x_train.shape)
print(y_train.shape)
import pdb;pdb.set_trace()

#print(y_train)
#import pdb;pdb.set_trace()
#(x_train, y_train), (x_test, y_test) = mnist.next_batch(12).load_data()
#print(y_train)
#pdb.set_trace()
x_train, x_test = x_train /255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)   

