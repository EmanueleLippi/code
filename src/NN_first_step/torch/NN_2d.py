import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# parameters
S0, K = 100.0, 100.0
r = 0.05
sigma = 0.2
T = 1.0
M = 50   # Numero di step temporali (ribilanciamo 50 volte)
N = 1024 # Numero di scenari (Batch size)
dt = T / M

class ZetaNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,32)
        self.layer2 = nn.Linear(32,32)
        self.output = nn.Linear(32,1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = torch.sigmoid(self.output(x))  # Usiamo sigmoid per mantenere Zeta in [0,1]
        return x
    
class BSDESolver(nn.Module):
    def __init__(self, S0, r, sigma, T, M):
        super().__init__()
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.M = M
        self.dt = T / M
        self.zeta_net = ZetaNetwork().to(device)
        self.Y0 = nn.Parameter(torch.tensor([[10.0]], device=device))  # Inizializziamo Y0 come parametro da ottimizzare
    
    def forward(self, dw, t_grid):
        S = torch.ones(dw.size(0), 1, device=device) * self.S0 # Stato iniziale S0
        Y = torch.ones(dw.size(0), 1, device=device) * self.Y0 # Stato iniziale Y0

        # Iteriamo nel tempo //TODO da ottimizzare
        for i in range(self.M):
            current_time = t_grid[:,i:i+1]
            net_input = torch.cat([S / self.S0, current_time], dim=1) # Normalizziamo S per stabilità
            zeta = self.zeta_net(net_input)
            dS = S * (self.r * self.dt + self.sigma * dw[:,i:i+1])
            interest_gain = self.r * (Y - zeta * S) * self.dt
            market_gain = zeta * dS

            Y = Y + interest_gain + market_gain
            S = S + dS
        return Y, S

solver = BSDESolver(S0, r, sigma, T, M).to(device)
optimizer = optim.Adam(solver.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
n_epochs = 2000
print(f"Prezzo Iniziale 'Ipotesi': {solver.Y0.item():.4f}")
loss_history = []
Y0_history = []

for epoch in range(2000):
    # Generiamo il "Caso" (Brownian Increments) al volo
    dW = torch.randn(N, M, device=device) * np.sqrt(dt)
    
    # Creiamo la griglia temporale normalizzata
    t_grid = torch.linspace(0, T, M+1, device=device)[:-1].repeat(N, 1) # [N, M]
    
    # Forward Pass attraverso tutto il tempo
    V_T_simulated, S_T_simulated = solver(dW, t_grid)
    
    # Calcoliamo il Payoff Reale (Target)
    payoff_target = torch.maximum(S_T_simulated - K, torch.tensor(0.0, device=device))
    
    # Loss: Vogliamo che il nostro portafoglio replichi esattamente il payoff
    loss = nn.MSELoss()(V_T_simulated, payoff_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    Y0_history.append(solver.Y0.item())
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Prezzo Stimato Y0: {solver.Y0.item():.4f}")

# --- 5. RISULTATI ---
# Prezzo teorico Black-Scholes (per confronto)
from scipy.stats import norm
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

print(f"\n--- RISULTATO FINALE ---")
print(f"Prezzo BSDE (Rete): {solver.Y0.item():.4f}")
print(f"Prezzo Black-Scholes (Esatto): {bs_price:.4f}")
print(f"Errore: {abs(solver.Y0.item() - bs_price):.4f}")

# Grafico convergenza Prezzo
plt.figure(figsize=(10,5))
plt.plot(Y0_history, label='Prezzo Stimato BSDE')
plt.axhline(y=bs_price, color='r', linestyle='--', label='Prezzo Black-Scholes')
plt.title("Convergenza del Prezzo dell'Opzione (Deep BSDE)")
plt.xlabel("Epoche")
plt.ylabel("Prezzo")
plt.legend()
plt.show()
