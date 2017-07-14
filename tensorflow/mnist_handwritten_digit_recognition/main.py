import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Setting up Tensorflow
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# Getting the dataset
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Method for visualizing a digit as image
def display(index):
	img = test_data[index]
	plt.title("Label {}".format(test_labels[index]))
	plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
	plt.show()

# Fit a Linear Classifier
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(n_classes=10, feature_columns=feature_columns)
classifier.fit(data, labels, batch_size=100, steps=1000)

# Evaluate accuracy
evl = classifier.evaluate(test_data, test_labels)
print("Accuracy: {}".format(evl['accuracy']))

# Get predictions for test data
pred = list(classifier.predict(test_data))

# Displaying
print("Predicted: {}, Label: {}".format(pred[1], test_labels[1]))
display(1)