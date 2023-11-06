from random import random
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, inputs, learning_rate):
        self.weights = []
        self.learning_rate = learning_rate
        for i in range(inputs):
            self.weights.append(random() * 2 + -1)

    def run(self, inputs):
        output = 0
        for i in range(len(self.weights)):
            output += self.weights[i] * inputs[i]
        return output

    def learn(self, inputs, desired):
        output = self.run(inputs)
        error = desired - output
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
        return error


if __name__ == "__main__":
    n = Neuron(2, 0.1)
    output = n.run([1, 2])
    errors = []
    for i in range(100):
        errors.append(n.learn([1, 2], 1.5))
    plt.plot(errors)
    plt.show()
