from simple_network import SimpleNetwork
from optimizer_SGD import Optimizer_SGD
from mse import mse_loss, mse_loss_derivative
import numpy as np
# Creiamo gli oggetti
network = SimpleNetwork()
optimizer = Optimizer_SGD(learning_rate=0.01) # Un learning rate più cauto

# Dati di training (Usiamo gli stessi per ora)
inputs = np.array([[120, 0.5, 0.2]])
targets = np.array([[25.0]]) # Supponiamo che il prezzo vero sia 25

# Il ciclo
for epoch in range(1000):
    # 1. Forward
    predictions = network.forward(inputs)
    
    # 2. Calcola gradiente loss
    loss_grad = mse_loss_derivative(predictions, targets)
    
    # 3. Backward (fai il passo indietro su layer2, poi su layer1)
    b1 = network.layer2.backward(loss_grad)
    b2 = network.layer1.backward(b1)
    
    # 4. Update (usa optimizer.update su entrambi i layer)
    optimizer.update(network.layer1)
    optimizer.update(network.layer2)
    
    # Ogni 100 epoche stampiamo l'errore per vedere se scende
    if epoch % 100 == 0:
        print(f"Epoca {epoch}, Loss: {mse_loss(predictions, targets)}")
print(f"Prezzo finale predetto: {network.forward(inputs)}")