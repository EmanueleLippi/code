import torch as tc
import numpy as np
import time
from abc import ABC, abstractmethod 
from typing import Tuple, List, Dict, Optional

class FBSDENN(tc.nn.Module, ABC):
    # M, N, D meglio che siano int
    def __init__(self, Xi: tc.Tensor, T: float, M: int, N: int, D: int, layers: List[int]):
        super().__init__()
        self.Xi = Xi  # punti di inizio
        self.T = T    # maturity
        self.M = M    # Numero traiettorie
        self.N = N    # Numero di step temporali
        self.D = D    # Numero di dimensioni
        self.layers = layers
        
        # Correzione check mps
        if tc.backends.mps.is_available():
            self.device = tc.device("mps")
        elif tc.cuda.is_available():
            self.device = tc.device("cuda")
        else:
            self.device = tc.device("cpu")
            
        # Importante: Xi e altri tensori salvati devono andare sul device
        # (Se Xi è un tensore, spostalo)
        if isinstance(self.Xi, tc.Tensor):
            self.Xi = self.Xi.to(self.device)
            
        # _createNN deve restituire nn.ParameterList per funzionare con .to(device)
        self.weights, self.biases = self._createNN(layers)
        
        # Questo sposterà automaticamente weights e biases sul device
        self.to(self.device)

    def _createNN(self, layers: List[int]) -> Tuple[tc.ParameterList, tc.ParameterList]:
        weights = tc.ParameterList()
        biases = tc.ParameterList()
        num_layers = len(layers)
        for l in range(num_layers - 1):
            W = tc.empty(layers[l], layers[l + 1])
            tc.nn.init.xavier_uniform_(W) # Inizializzazione Xavier sostituisce self.xavier_init
            b = tc.zeros(1, layers[l + 1])
            weights.append(tc.nn.Parameter(W))
            biases.append(tc.nn.Parameter(b))
        return weights, biases
    
    def neural_net(self, X: tc.Tensor, weights, biases) -> tc.Tensor:
        num_layers = len(weights) + 1
        H = X 
        
        # Hidden layers (fino al penultimo)
        for l in range(num_layers - 2):
            W = weights[l]
            b = biases[l]
            # Importante: aggiorna H, non creare una nuova variabile Y qui
            H = tc.sin(tc.add(tc.matmul(H, W), b))
        
        # Output layer (l'ultimo layer, fuori dal ciclo, lineare senza sin)
        W = weights[-1]
        b = biases[-1]
        Y = tc.add(tc.matmul(H, W), b)
        
        return Y

    def net_u(self, t: tc.tensor, X: tc.tensor):
        input_data = tc.cat((t, X), dim=1)
        u = self.neural_net(input_data, self.weights, self.biases)
        #create_graph=True è necessario perché Du viene usato nella loss (second backprop)
        #retain_graph=True spesso utile per sicurezza se chiami grad più volte
        Du = tc.autograd.grad(output=u, inputs=X, grad_outputs=tc.ones_like(u), create_graph=True, retain_graph=True)[0]
        return u, Du
    
    def Dg_tf(self, X: tc.tensor) --> tc.tensor:
        #calcolo g(x)
        g = self._g_tf(X)
        #calcolo dg/dx
        Dg = tc.autograd.grad(output=g, input=X, grad_outputs=tc.ones_like(g), create_graph=True)[0]
        return Dg

