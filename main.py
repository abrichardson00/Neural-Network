import numpy as np
import random
from NN import NN
from Neuron import Neuron
from NN_Display import NN_Display

### just some test training data of the form [x,x,x,x,x,x]
### with an output of [1,0] if the sum of all the xs is less
### than 3 and an output of [0,1] if the sum is more than 3
training_data = []
for i in range(1000):
    x = np.zeros((6))
    sum = 0
    for j in range(6):
        x[j] = random.random()
        sum += x[j]
    if sum < 3:
        output = np.array([1,0])
    else:
        output = np.array([0,1])
    training_data.append((x,output))



### can either initialize the network from scratch of from a file
network = NN(6,[4,2])
#network = NN.loadFromFile('network1.txt')

### displaying the network
network_display = NN_Display(800,800)
network_display.drawNeurons(network)
network_display.drawWeights(network)

### training the network
network.SGD(training_data,20,20,10,network_display)
network_display.drawWeights(network)

### some test examples
### hopefully the output should be close to [1,0] or [0,1] depending on the input
input = [1,1,1,1,1,1]
print("Input: " + str(input))
print("Output: " + str(network.evaluateInput(np.array(input))))
input = [1,1,1,0,0,1]
print("Input: " + str(input))
print("Output: " + str(network.evaluateInput(np.array(input))))
input = [0.2,0.5,0,0,0,1]
print("Input: " + str(input))
print("Output: " + str(network.evaluateInput(np.array(input))))
network_display.holdDisplay()

### do we want to save the network?
network.saveToFile('network1')
