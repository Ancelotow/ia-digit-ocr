import matplotlib.pyplot as plt
import numpy as np
from classifier import Classifier

def get_data():
    data = np.loadtxt('dataset.csv', skiprows=1, delimiter=',')
    inputs = data[:, :- 1] / 255
    outputs = data[:, -1]
    return inputs, outputs


def one_hot_encoder(value, values):
    return [1 if x == value else 0 for x in values]


if __name__ == '__main__':
    inputs, outputs = get_data()
    classifier = Classifier(len(inputs[0]), 10, 1E-3)
    errors = []
    for j in range(300):
        for i in range(len(inputs)):
            desired = one_hot_encoder(outputs[i], list(range(10)))
            errors.append(classifier.learn(inputs[i],desired))
    plt.plot(errors)
    plt.show()

    error = 0.0
    predictions = []
    for i in range(len(inputs)):
        prediction = classifier.run(inputs[i])
        print(prediction, ' -- ', outputs[i])

