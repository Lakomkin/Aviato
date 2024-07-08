import os
import pandas as pd 
import tensorflow as tf

import numpy as np

train_routes = np.load('datasets/train_routes.npy')
train_labels = np.load('datasets/train_labels.npy')
test_routes = np.load('datasets/test_routes.npy')
test_labels = np.load('datasets/test_labels.npy')

print(train_routes)
print(train_labels)
#print(test_routes)
print(test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_routes, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_routes,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)