from percepron import DenseLayer
import numpy as np

class SimpleNetwork:
    def __init__(self):
        self.layer1 = DenseLayer(n_inputs=3, n_neurons=4, activation='relu')
        self.layer2 = DenseLayer(n_inputs=4, n_neurons=1, activation=None)
    
    def forward(self, inputs):
        #passa i dati x attraverso il primo layer
        out1 = self.layer1.forward(inputs)
        #passa i dati output del primo layer attraverso il secondo layer
        out2 = self.layer2.forward(out1)

        return out2
    
#Test 
#network = SimpleNetwork()
#inputs = np.array([[120, 0.5, 0.2]])
#print(f"Input: {inputs}")
#print(f"Output: {network.forward(inputs)}")