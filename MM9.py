'''
Deep learning of MNIST dataset using TensorFlow.
Running on a TensorFlow docker image.
'''
import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from _overused_functions import Overused

data = loadmat('MM4_material/mnist_all.mat')

train_data, train_labels, test_data, test_labels, accuracy_target_classes = Overused().to_sep(data, to_shuffle=True)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(train_data[:1])
#tf.nn.softmax(predictions)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss_fn(train_labels[:1], predictions)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=5)
prediction_labels = model.predict(test_data)
to_softmax = tf.nn.softmax(prediction_labels)
# Getting the labels

prediction_labels = np.asarray([np.argmax(to_softmax[i]) for i in range(len(prediction_labels))])
# Evaluating the model
model.evaluate(test_data, test_labels, verbose=2)

# Computing the accuracy
accuracy = [np.sum(prediction_labels[test_labels == i] == i) / len(accuracy_target_classes[i]) * 100 for i in range(10)]
print(accuracy)