import matplotlib.pyplot as plt
from neuron import Neuron


def get_data():
    inputs = []
    outputs = []
    with open("dataset.csv") as f:
        for line in f.readlines()[1:]:
            image = list(map(int, line.split(',')))
            inputs.append(image[:783])
            outputs.append(image[784])
    return inputs, outputs


def classification(inputs, outputs):
    n = Neuron(len(inputs[0]), 1E-7)
    errors = []
    for j in range(3000):
        for i in range(len(inputs)):
            errors.append(n.learn(inputs[i], outputs[i]))

    plt.plot(errors)
    plt.show()

    error = 0.0
    predictions = []
    for i in range(len(inputs)):
        prediction = n.run(inputs[i])
        print(prediction, ' -- ', outputs[i])
        predictions.append(prediction)
        error += abs(prediction - outputs[i])
    error = error / len(inputs)
    return n, predictions, error

def show_digit_vector(vector):
    fig, ax = plt.subplot()
    ax.matshow(vector.reshape)

if __name__ == '__main__':
    inputs, outputs = get_data()
    reg, predictions, error = classification(inputs, outputs)
    print(f'error={error}')

