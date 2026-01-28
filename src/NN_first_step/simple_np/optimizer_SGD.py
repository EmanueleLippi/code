class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        
    def update(self, layer):
        # Applica la formula di aggiornamento ai pesi e ai bias del layer
        layer.weight = layer.weight - self.learning_rate * layer.dw
        layer.bias = layer.bias - self.learning_rate * layer.db