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


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


def load_data():
    print 'Loading data...'
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))

    print 'Data loaded.'
    return [X_train, X_test, y_train, y_test]


def init_model():
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(Dense(500, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    print 'Model compield in {0} seconds'.format(time.time() - start_time)
    return model

# Below if a modified version of the function from the tutorial.
def run_my_network(model, X_train, X_test,  y_train, y_test, epochs=20, batch=256):
    try:
        start_time = time.time()
        history = LossHistory()
        print 'Training model...'
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history], show_accuracy=True,
                  validation_data=(X_test, y_test), verbose=2)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_test, batch_size=16,
                               show_accuracy=True)
        print(type(score))
        print(score)
        print "Network's test score [loss, accuracy]: {0}".format(score)
        return model, score,  history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, score, history.losses


def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()
    plt.show()

############## START THE MAIN PROGRAM WITH BAGGING #############

# The following function implements the bagging algorithm. This is the main function in this program.
# In Bagging we first do bootstrap sampling to randomly select a subset of the dataset. We run our model on
# this subset and get the score. We repeat this process for a number of iterations and then average the losses over
# those iterations.
# ARGS:
# numberOfIterations: number of iterations (When 1 defaults to full sample size without bagging)
#  sampleSize: Sample size for each iterations.
# X_train:
# X_test:
# y_train:
# y_test:
def run_bagging_algo(model, X_train, X_test, y_train, y_test, numberOfIterations, sampleSize):
    total_score = 0
    # losses_array= np.array([])
    models_list = []
    for i in range(0, numberOfIterations):
        print('!!!!!!!!!!Iteration!!!!!!!!!!!')
        # If numberOfIterations = 1, use full sample size (No Bagging)
        if(numberOfIterations == 1):
            X_train_sample=X_train
            y_train_sample=y_train
        else:
            index=npr.randint(0, 60000, (1,sampleSize))
            X_train_temp=X_train[index]
            X_train_sample=X_train_temp[0]
            y_train_temp=y_train[index]
            y_train_sample=y_train_temp[0]
        model, score, losses = run_my_network(model, X_train_sample, X_test, y_train_sample, y_test)
        total_score = total_score + score
        if(i==0):
            print('-------------- IF LOSSES ARRAY -------------')
            losses_array = np.array(losses)
            print(losses_array)
            print(losses_array.shape)
        else:
            losses_array= np.vstack([losses_array,losses])
            print('-------------- ELSE LOSSES ARRAY -------------')
            print(losses_array)
            print(losses_array.shape)
        # losses_list.append(losses)
        models_list.append(model)
    if(numberOfIterations==1):
        average_score = total_score
        average_losses = losses_array
    else:
        average_score = total_score/numberOfIterations
        average_losses =  np.nanmean(losses_array, axis=0)
    print('------- average_losses --')
    print(average_losses)
    return models_list, average_score, average_losses

plt.interactive(False) # Set to work in Pycharm IDE
#Load the MNIST data set
X_train, X_test, y_train, y_test = load_data()
model = init_model()

# When numberOfIterations = 1, the full sample will be used. This defaults to WITHOUT Bagging.
# We fist find the score and losses without Bagging (i.e. numberOfIterations = 1.
numberOfIterations = 1
sampleSize = 40000
models_list, average_score, average_losses = run_bagging_algo(model, X_train, X_test, y_train, y_test, numberOfIterations, sampleSize)
print('------- average_score -------')
print(average_score)
print('------- average_losses --')
print(average_losses)
plot_losses(average_losses)

# The following code will do Bagging with bootstrapped sample size of 40000 with 14 iterations.
numberOfIterations = 14
sampleSize = 40000
models_list, average_score, average_losses = run_bagging_algo(model, X_train, X_test, y_train, y_test, numberOfIterations, sampleSize)
print('------- average_score -------')
print(average_score)
print('------- average_losses --')
print(average_losses)
plot_losses(average_losses)

