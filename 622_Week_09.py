
import time
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from sklearn import datasets, svm, metrics

digitData = datasets.load_digits()
n_samples = len(digitData.images)
data = digitData.images.reshape((n_samples, -1))
print("The number of total samples is " + str(n_samples))
# Create a SVN classifier.
svm_classifier = svm.SVC(gamma=0.001)

# Train the model on first half of the digitData.
svm_classifier.fit(data[:n_samples / 2], digitData.target[:n_samples / 2])

# Get the exepected digits before prediction.
expected = digitData.target[n_samples / 2:]

# Predict the values on the second half of the digits.
predicted = svm_classifier.predict(data[n_samples / 2:])

# Compare the predicted to the expected and get the percentage error.
errors=0
for i in range(predicted.size):
    if predicted[i] != expected[i]:
        errors += 1
print("The number of wrongly predicted values is " + str(errors) + " out of " + str(predicted.size))

# Print out the percentage error.
percentageError = (errors/float(predicted.size))*100
print ("The percentage prediction error is " + str( round(percentageError, 2)) + "%")