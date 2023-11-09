import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from classifier import Classifier
from os.path import exists

FILE_MODEL = "model.dat"
LEARNING_RATE = 1E-3
NB_ITERATION = 1
DATASET_NAME = "dataset-full.csv"


def get_data():
    data = np.loadtxt(DATASET_NAME, skiprows=1, delimiter=',')
    inputs = data[:, :- 1] / 255
    outputs = data[:, -1]
    return inputs, outputs


def one_hot_encoder(value, values):
    return [1 if x == value else 0 for x in values]


def learning(classifier, inputs, outputs):
    errors = []
    for j in range(NB_ITERATION):
        for i in range(len(inputs)):
            desired = one_hot_encoder(outputs[i], list(range(10)))
            errors.append(classifier.learn(inputs[i], desired))
    plt.plot(errors)
    plt.show()


if __name__ == '__main__':
    inputs, outputs = get_data()
    if not exists(FILE_MODEL):
        classifier = Classifier(len(inputs[0]), 10, LEARNING_RATE)
        learning(classifier, inputs, outputs)
        with open(FILE_MODEL, 'wb') as file:
            pickle.dump(classifier, file)
    else:
        with open(FILE_MODEL, 'rb') as file:
            classifier = pickle.load(file)

    predictions = []
    for i in range(len(inputs)):
        predictions.append(classifier.run(inputs[i]))
    print(confusion_matrix(outputs, predictions))