from src import network
from src import mnist_loader
from PIL import Image
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#img_data = [Image.fromarray(t[0].reshape(28, 28) * 255).convert("L").save(f"data/{i}_{t[1].argmax()}.png") for i, t in enumerate(training_data[:100])]

w = np.fromfile("weights", dtype=np.float32)
weights = [w[:784*30], w[784*30:784*30+30*10]]
assert weights[0][0] == w[0] and weights[-1][-1] == w[-1] and weights[0][-1] != weights[-1][0] and len(weights[0]) + len(weights[1]) == len(w)
weights = [weights[0].reshape(30, 784), weights[1].reshape(10, 30)]

b = np.fromfile("biases", dtype=np.float32)
biases = [b[:30], b[30:30+10]]
assert biases[0][0] == b[0] and biases[-1][-1] == b[-1] and biases[0][-1] != biases[-1][0] and len(biases[0]) + len(biases[1]) == len(b)
biases = [biases[0].reshape(30, 1), biases[1].reshape(10, 1)]

net = network.Network([784, 30, 10])
net.weights = weights
net.biases = biases
net.SGD(training_data, 1, 10, 3.0, test_data=test_data)
print()
b.tofile()