import numpy as np
def mse_loss(prediction, target):
    delta = prediction-target
    delta_sqr = np.power(delta, 2)
    return np.mean(delta_sqr)

def mse_loss_derivative(predictions, targets):
    # N = numero di campioni (puoi usare targets.size o targets.shape[0])
    N = targets.size
    # Formula: 2 * (differenza) / N
    grad = 2 * (predictions - targets) / N
    return grad

#Test
#prediction = np.array([[0.1, 0.2, 0.7]])
#target = np.array([[1, 0, 0]])
#print(mse_loss(prediction, target))