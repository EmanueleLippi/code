import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

# --- CONFIGURAZIONE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Parametri Finanziari
DIM = 5          # 5 Azioni nel paniere
S0_val = 100.0   # Prezzo iniziale per tutte le azioni
K = 100.0        # Strike sulla MEDIA dei prezzi
r = 0.05
sigma = 0.2
T = 1.0

# Parametri Training
M = 50           # Step temporali
N = 4096         # Batch size
EPOCHS = 3000    # Un po' più epoche perché il problema è più complesso
LEARNING_RATE = 0.005 # LR leggermente più basso per stabilità

dt = T / M

# --- RETE NEURALE MULTIDIMENSIONALE ---
class ZetaNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, hidden_dim), # Layer extra per complessità 5D
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, output_dim), # Output: 5 Delta (uno per asset)
            nn.Sigmoid() # Assumiamo Delta positivi per una Call
        )
    
    def forward(self, x):
        return self.net(x)

# --- SOLVER BSDE 5D ---
class BasketBSDESolver(nn.Module):
    def __init__(self, dim, S0, r, sigma, T, M, device):
        super().__init__()
        self.dim = dim
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.M = M
        self.dt = T / M
        self.device = device
        
        # Input: 5 prezzi + 1 tempo = 6 input
        # Output: 5 delta
        self.zeta_net = ZetaNetwork(input_dim=dim+1, output_dim=dim).to(device)
        
        # Prezzo iniziale dell'opzione (parametro da imparare)
        self.Y0 = nn.Parameter(torch.tensor([10.0], device=device)) 
    
    def forward(self, dw, t_grid):
        # dw: [N, M, DIM] - Incrementi Browniani Indipendenti (per semplicità)
        batch_size = dw.size(0)
        
        # S parte come matrice [N, DIM] tutta a 100.0
        S = torch.ones(batch_size, self.dim, device=self.device) * self.S0
        Y = torch.ones(batch_size, 1, device=self.device) * self.Y0
        
        for i in range(self.M):
            # Input rete: Prezzi normalizzati + Tempo
            # S/S0 -> [N, DIM], time -> [N, 1] => Concateniamo a [N, DIM+1]
            current_time = t_grid[:, i:i+1]
            net_input = torch.cat([S / self.S0, current_time], dim=1) 
            
            # La rete ci dà i 5 Delta: [N, DIM]
            zeta = self.zeta_net(net_input)
            
            # Dinamica Asset (Vettorizzata)
            # dw[:, i, :] estrae i random shock per tutti e 5 gli asset allo step i
            dw_step = dw[:, i, :] 
            dS = S * (self.r * self.dt + self.sigma * dw_step)
            
            # Dinamica Portafoglio (Hedging su 5 asset)
            # Guadagno Asset = Somma(Delta_k * dS_k) su k=1..5
            # torch.sum(..., dim=1) somma lungo le colonne (gli asset)
            portfolio_gain = torch.sum(zeta * dS, dim=1, keepdim=True)
            
            # Costo portafoglio azioni = Somma(Delta_k * S_k)
            asset_holdings_value = torch.sum(zeta * S, dim=1, keepdim=True)
            
            # Evoluzione Y
            interest_gain = self.r * (Y - asset_holdings_value) * self.dt
            Y = Y + interest_gain + portfolio_gain
            
            # Aggiorniamo S per il prossimo step
            S = S + dS
            
        return Y, S

def train_basket():
    print(f"--- Inizio Training Basket Option ({DIM} Assets) ---")
    
    solver = BasketBSDESolver(DIM, S0_val, r, sigma, T, M, device).to(device)
    
    # JIT Compilation
    try:
        solver_scripted = torch.jit.script(solver)
        print("JIT Compilation: SUCCESSO 🚀")
    except Exception as e:
        print(f"JIT Compilation: FALLITO ({e}) - Procedo standard")
        solver_scripted = solver

    optimizer = optim.Adam(solver_scripted.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    loss_history = []
    Y0_history = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Generiamo dw con dimensione extra per gli asset: [N, M, DIM]
        dW = torch.randn(N, M, DIM, device=device) * np.sqrt(dt)
        t_grid = torch.linspace(0, T, M+1, device=device)[:-1].repeat(N, 1)
        
        V_T, S_T = solver_scripted(dW, t_grid)
        
        # --- PAYOFF BASKET ARITMETICO ---
        # Media dei prezzi finali dei 5 asset
        basket_mean = torch.mean(S_T, dim=1, keepdim=True)
        payoff_target = torch.maximum(basket_mean - K, torch.tensor(0.0, device=device))
        
        loss = loss_fn(V_T, payoff_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        Y0_history.append(solver.Y0.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Y0: {solver.Y0.item():.4f}")

    total_time = time.time() - start_time
    print(f"--- Training Completato in {total_time:.2f}s ---")
    
    return solver, Y0_history

# --- BENCHMARK: MONTE CARLO CLASSICO ---
def monte_carlo_basket_price(S0, K, r, sigma, T, dim, n_sims=500000):
    print(f"\nCalcolo Monte Carlo Benchmark ({n_sims} simulazioni)...")
    # Generiamo solo il prezzo finale S_T direttamente
    # S_T = S0 * exp(...)
    Z = np.random.normal(0, 1, (n_sims, dim))
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Media del paniere
    basket_avg = np.mean(S_T, axis=1)
    payoffs = np.maximum(basket_avg - K, 0)
    
    # Scontiamo a oggi
    price = np.mean(payoffs) * np.exp(-r * T)
    return price

if __name__ == "__main__":
    solver, Y0_hist = train_basket()
    
    bsde_price = solver.Y0.item()
    mc_price = monte_carlo_basket_price(S0_val, K, r, sigma, T, DIM)
    
    print(f"\n--- RISULTATI BASKET OPTION (5D) ---")
    print(f"BSDE Price (Rete):  {bsde_price:.4f}")
    print(f"Monte Carlo Price:  {mc_price:.4f}")
    print(f"Differenza:         {abs(bsde_price - mc_price):.4f}")
    
    plt.figure(figsize=(10,5))
    plt.plot(Y0_hist, label='BSDE Price')
    plt.axhline(y=mc_price, color='r', linestyle='--', label=f'Monte Carlo ({mc_price:.2f})')
    plt.title(f"Pricing Basket Option {DIM}-Dim (Deep BSDE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()