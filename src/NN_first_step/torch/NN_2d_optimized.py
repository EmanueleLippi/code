import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import norm

# --- CONFIGURAZIONE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Parametri Finanziari
S0, K = 100.0, 100.0
r = 0.05
sigma = 0.2
T = 1.0

# Parametri Training
M = 50          # Step temporali
N = 4096        # Batch size (Aumentato per efficienza su GPU/MPS)
EPOCHS = 2000
LEARNING_RATE = 0.01

dt = T / M

# --- RETE NEURALE (Migliorata) ---
class ZetaNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Normalizzazione per stabilità
            nn.GELU(),                # Attivazione moderna (Gaussian Error Linear Unit)
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()              # Mantiene l'output in [0, 1] come nel paper originale
        )
    
    def forward(self, x):
        return self.net(x)

# --- SOLVER BSDE (JIT COMPATIBILE) ---
# Per usare torch.jit.script, dobbiamo evitare strutture dinamiche Python nel loop.
# Definiamo il modulo in modo che sia "statico".

class BSDESolver(nn.Module):
    def __init__(self, S0: float, r: float, sigma: float, T: float, M: int, device: torch.device):
        super().__init__()
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.M = M
        self.dt = T / M
        self.device = device
        
        self.zeta_net = ZetaNetwork().to(device)
        self.Y0 = nn.Parameter(torch.tensor([10.0], device=device)) 
    
    def forward(self, dw: torch.Tensor, t_grid: torch.Tensor):
        # dw: [N, M] - Incrementi Browniani
        # t_grid: [N, M] - Griglia temporale
        
        batch_size = dw.size(0) #ottieni il numero di batch come dimensione del primo tensor
        
        # Inizializzazione stati
        S = torch.ones(batch_size, 1, device=self.device) * self.S0
        Y = torch.ones(batch_size, 1, device=self.device) * self.Y0
        
        # Loop temporale compilabile
        for i in range(self.M):
            # Preparazione input rete: [S_norm, time]
            current_time = t_grid[:, i:i+1]
            # Normalizziamo S dividendo per S0 (comune in finanza)
            net_input = torch.cat([S / self.S0, current_time], dim=1) 
            
            # Calcolo strategia Zeta (Delta-hedging approssimato)
            zeta = self.zeta_net(net_input)
            
            # Dinamica S (Geometric Brownian Motion)
            # dS = S * (r dt + sigma dW)
            dS = S * (self.r * self.dt + self.sigma * dw[:, i:i+1])
            
            # Dinamica Y (Portafoglio replicante)
            # dY = r(Y - zeta*S)dt + zeta*dS
            interest_gain = self.r * (Y - zeta * S) * self.dt
            market_gain = zeta * dS
            
            Y = Y + interest_gain + market_gain
            S = S + dS
            
        return Y, S

# Scripting del modello per ottimizzazione JIT
# Nota: Creiamo l'istanza e poi applichiamo lo script, oppure usiamo il decoratore se la classe è semplice.
# Qui facciamo lo scripting dell'istanza per flessibilità.

def train():
    print("--- Inizio Training Ottimizzato ---")
    
    # Istanziazione
    solver = BSDESolver(S0, r, sigma, T, M, device).to(device)
    
    # Tentativo di JIT Scripting
    # Passiamo input dummy per tracciare o compilare
    try:
        # Scripting è preferibile a Tracing per loop di controllo
        solver_scripted = torch.jit.script(solver)
        print("JIT Compilation: SUCCESSO")
    except Exception as e:
        print(f"JIT Compilation: FALLITO ({e}) - Procedo con Python standard")
        solver_scripted = solver

    optimizer = optim.Adam(solver_scripted.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    loss_fn = nn.MSELoss()
    
    loss_history = []
    Y0_history = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Generazione dati on-the-fly (molto veloce su GPU/MPS)
        dW = torch.randn(N, M, device=device) * np.sqrt(dt)
        t_grid = torch.linspace(0, T, M+1, device=device)[:-1].repeat(N, 1)
        
        # Forward
        V_T_simulated, S_T_simulated = solver_scripted(dW, t_grid)
        
        # Target Payoff
        payoff_target = torch.maximum(S_T_simulated - K, torch.tensor(0.0, device=device))
        
        # Loss
        loss = loss_fn(V_T_simulated, payoff_target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        loss_history.append(loss.item())
        Y0_history.append(solver.Y0.item()) # Accediamo al parametro originale
        
        if epoch % 200 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Y0: {solver.Y0.item():.4f} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"--- Training Completato in {total_time:.2f}s ---")
    
    return solver, loss_history, Y0_history

# --- CALCOLO ESATTO B&S ---
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

if __name__ == "__main__":
    solver, loss_hist, Y0_hist = train()
    
    final_price = solver.Y0.item()
    print(f"\n--- RISULTATI ---")
    print(f"BSDE Price: {final_price:.4f}")
    print(f"Black-Scholes Price: {bs_price:.4f}")
    print(f"Error: {abs(final_price - bs_price):.4f}")
    
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(Y0_hist, label='BSDE Price')
    plt.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes')
    plt.title(f"Ottimizzazione BSDE (M={M}, N={N})")
    plt.xlabel("Epochs")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
