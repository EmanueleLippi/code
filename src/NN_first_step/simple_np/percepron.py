import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        self.weight = 0.01 * np.random.randn(n_inputs, n_neurons) #matrice dei pesi per la rete neurale di dimensione (n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons)) #vettore dei bias per la rete neurale di dimensione (1, n_neurons)
        self.inputs = None #contiene gli input passati al layer
        self.z = None #contiene il prodotto scalare tra gli input e i pesi e aggiunge il bias
        self.dw = None #contiene il gradiente dei pesi
        self.db = None #contiene il gradiente dei bias
        self.dinputs = None #contiene il gradiente degli input
        self.activation = activation
    
    def forward(self, inputs):
        #calcola il prodotto scalare tra gli input e i pesi e aggiunge il bias
        z = np.dot(inputs, self.weight) + self.bias
        #applica la funzione di attivazione
        if self.activation == 'relu':
            output = np.maximum(0, z)
        else: # Linear
            output = z
        self.inputs = inputs #salva gli input per il backpropagation
        self.z = z #salva il prodotto scalare per il backpropagation
        return output

    def backward(self, dvalues):
        #calcola il gradiente dei pesi e dei bias
        if self.activation == 'relu':
            dvalues[self.z < 0] = 0
        # per linear, la derivata è 1, quindi dvalues resta invariato
        
        self.dw = np.dot(self.inputs.T, dvalues)
        self.db = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weight.T)
        return self.dinputs

#layer = DenseLayer(n_inputs=3, n_neurons=4)
#inputs = np.array([[120, 0.5, 0.2]]) # Nota le doppie parentesi per creare un vettore riga (1,3)
#print(f"Input: {inputs}")
#print(f"Output del layer: {layer.forward(inputs)}")
#print(f"Gradiente degli input: {layer.backward(inputs)}")