import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# --- CONFIGURAZIONE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Benchmark Longstaff-Schwartz (2001) - Case 1
S0 = 36.0
K = 40.0
r = 0.06
sigma = 0.2
T = 1.0

# Parametri Simulazione
M = 50        
N = 50000     
dt = T / M

# --- GENERATORE PERCORSI ---
def generate_paths(S0, r, sigma, T, M, N, device):
    dt = T / M
    S = torch.zeros(N, M + 1, device=device)
    S[:, 0] = S0
    Z = torch.randn(N, M, device=device)
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    log_returns = torch.cumsum(drift + vol * Z, dim=1)
    S[:, 1:] = S0 * torch.exp(log_returns)
    return S

# --- RETE NEURALE PERSISTENTE ---
class ContinuationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# --- POLYNOMIAL REGRESSION (FASTEST) ---
def polynomial_regression(X_norm, Y, degree=3):
    # X_norm: [N, 1], Y: [N, 1]
    # Costruiamo la matrice delle features [1, x, x^2, x^3]
    # Nota: X_norm è già normalizzato (S/K)
    features = [torch.ones_like(X_norm)]
    for d in range(1, degree + 1):
        features.append(torch.pow(X_norm, d))
    
    A = torch.cat(features, dim=1) # [N, degree+1]
    
    # Spostiamo su CPU perchè lstsq non è implementato su MPS
    A_cpu = A.cpu()
    Y_cpu = Y.cpu()
    
    # Risolviamo su CPU
    solution = torch.linalg.lstsq(A_cpu, Y_cpu).solution
    
    # Ritorniamo su device originale
    return (A @ solution.to(X_norm.device))

# --- DEEP LSM ALGORITHM (OPTIMIZED) ---
def run_optimized_lsm(mode="neural"):
    print(f"--- Generazione {N} Percorsi ---")
    paths = generate_paths(S0, r, sigma, T, M, N, device)
    
    S_T = paths[:, -1]
    cashflows = torch.maximum(K - S_T, torch.tensor(0.0, device=device))
    
    discount_factor = np.exp(-r * dt)
    
    print(f"--- Inizio Backward Induction ({mode.upper()} Mode) ---")
    start_time = time.time()
    
    # INIZIALIZZAZIONE UNICA DEL MODELLO (WARM START)
    if mode == "neural":
        model = ContinuationNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005) # LR leggermente più basso per stabilità
        loss_fn = nn.MSELoss()
        
        # Scheduler per ridurre il learning rate man mano che ci avviciniamo a t=0? 
        # Non necessariamente utile qui, perché il problema cambia.
    
    # Loop all'indietro
    for t in range(M - 1, 0, -1):
        cashflows = cashflows * discount_factor
        S_t = paths[:, t]
        
        itm_mask = S_t < K 
        num_itm = torch.sum(itm_mask).item()
        
        if num_itm < 100:
            continue
            
        X_itm = S_t[itm_mask].reshape(-1, 1)
        Y_itm = cashflows[itm_mask].reshape(-1, 1)
        X_norm = X_itm / K
        
        # PREDIZIONE CONTINUATION VALUE
        if mode == "neural":
            # Training incrementale (pochi step)
            model.train()
            EPOCHS_PER_STEP = 5 # Sufficienti se partiamo dai pesi precedenti
            
            for _ in range(EPOCHS_PER_STEP):
                pred = model(X_norm)
                loss = loss_fn(pred, Y_itm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                continuation_value = model(X_norm).flatten()
                
        elif mode == "poly":
            # Regressione polinomiale diretta
            continuation_value = polynomial_regression(X_norm, Y_itm, degree=4).flatten()
        
        # DECISIONE ESERCIZIO
        exercise_value = K - X_itm.flatten()
        should_exercise = exercise_value > continuation_value
        
        # AGGIORNAMENTO CASHFLOW
        global_indices = torch.nonzero(itm_mask).flatten()
        exercise_indices = global_indices[should_exercise]
        cashflows[exercise_indices] = exercise_value[should_exercise]

        if t % 10 == 0:
            print(f"Step {t}/{M}")

    # Step finale
    cashflows = cashflows * discount_factor
    price = torch.mean(cashflows).item()
    
    elapsed = time.time() - start_time
    print(f"Tempo calcolo ({mode}): {elapsed:.2f}s")
    return price, elapsed

if __name__ == "__main__":
    true_price = 4.478
    
    print("\n============== NEURAL MODE (WARM START) ==============")
    price_nn, time_nn = run_optimized_lsm(mode="neural")
    print(f"Price: {price_nn:.4f} | Error: {abs(price_nn - true_price):.4f} | Time: {time_nn:.2f}s")
    
    print("\n============== POLYNOMIAL MODE (FAST) ==============")
    price_poly, time_poly = run_optimized_lsm(mode="poly")
    print(f"Price: {price_poly:.4f} | Error: {abs(price_poly - true_price):.4f} | Time: {time_poly:.2f}s")
