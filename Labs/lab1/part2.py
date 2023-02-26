import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalHinge
from tensorflow.keras.regularizers import l1, l2, l1_l2


def combinationMaker(activation, regularizer, optimizer, loss, initializer, initializerName):

    dataset = pd.read_csv(
        "https://raw.githubusercontent.com/KhaledElTahan/DeepLearning/master/Labs/lab1/lab1_heart.csv")

    X = dataset.iloc[:, 0:13].values
    y = dataset.iloc[:, 13].values

    x = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # Get Training Data
    train_X, temporary_X, train_y, temporary_y = train_test_split(
        X, y, train_size=0.75, random_state=0)

    # Get Validation & Testing Data
    val_X, test_X, val_y, test_y = train_test_split(
        temporary_X, temporary_y, train_size=0.5, random_state=0)

    model = Sequential()
    model.add(tf.keras.Input(shape=(train_X.shape[1],)))

    model.add(Dense(1, activation=activation,
              kernel_regularizer=regularizer(0.01), kernel_initializer=initializer))

    # TODO Try Different losses & optimizers here
    model.compile(loss=loss(), metrics=[
        'accuracy'], optimizer=optimizer())
    # model.summary()

    hist = model.fit(train_X, train_y, verbose=1, validation_data=(
        val_X, val_y), batch_size=16, epochs=500)

    score, accuracy = model.evaluate(test_X, test_y, batch_size=16, verbose=0)
    print("Test fraction correct (NN-Loss) = {:.2f}".format(score))
    print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))

    # Get training and test loss histories
    training_loss = hist.history['accuracy']
    val_loss = hist.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.figure()
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.show()
    # save the figure as a png file in the ./Screenshots_part2 folder with the name of the combination
    plt.savefig(
        f"./screenshots_part2/{activation}_{regularizer.__name__}_{optimizer.__name__}_{loss.__name__}_{initializerName}.png")


if __name__ == "__main__":
    #  list of activation functions as strings
    activations = ['sigmoid', 'tanh', 'relu',
                   'elu', 'selu', 'softplus', 'softsign']

    #  list of regularizers as functions
    regularizers = [l1, l2]

    #  list of optimizers as functions
    optimizers = [Adam, SGD, Adagrad, RMSprop]

    #  list of losses as functions
    losses = [BinaryCrossentropy, CategoricalHinge]

    #  list of initializers as functions
    initializers = [RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    RandomUniform(minval=-0.05, maxval=0.05, seed=None)]
    initializersNames = ['RandomNormal', 'RandomUniform']

    for activation in activations:
        for regularizer in regularizers:
            for optimizer in optimizers:
                for loss in losses:
                    for initializer in initializers:
                        combinationMaker(activation, regularizer,
                                         optimizer, loss, initializer ,initializersNames[initializers.index(initializer)])
