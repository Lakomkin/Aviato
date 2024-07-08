from pickletools import optimize
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt 


train_x = np.load('datasets/train_routes.npy')
train_y = np.load('datasets/train_labels.npy')
test_x = np.load('datasets/test_routes.npy')
test_y = np.load('datasets/test_labels.npy')

 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) 
model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax)) 

 
model.compile(  optimizer = 'adam', 
                loss= 'sparse_categorical_crossentropy', 
                metrics=['accuracy'])

model.fit(train_x, train_y, batch_size = 1, epochs = 10)

val_loss, val_acc = model.evaluate(test_x, test_y)

print(val_loss, val_acc)

predictions = model.predict([test_x])


print(np.argmax(predictions[200]))
plt.imshow(train_x[200])
plt.show()