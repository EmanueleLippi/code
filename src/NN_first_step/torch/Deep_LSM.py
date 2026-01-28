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

# Parametri Simulazione POTENZIATI
M = 50        # Step temporali
N = 50000     # AUMENTATO: 50k percorsi riducono drasticamente la varianza
dt = T / M

# --- GENERATORE PERCORSI ---
def generate_paths(S0, r, sigma, T, M, N, device):
    # Generazione vettorizzata su GPU
    dt = T / M
    S = torch.zeros(N, M + 1, device=device)
    S[:, 0] = S0
    Z = torch.randn(N, M, device=device)
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    log_returns = torch.cumsum(drift + vol * Z, dim=1)
    S[:, 1:] = S0 * torch.exp(log_returns)
    return S

# --- RETE NEURALE ---
# Più capacità (64 neuroni) per fittare meglio la curva di continuazione
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

# --- DEEP LSM ALGORITHM ---
def run_deep_lsm_turbo():
    print(f"--- Generazione {N} Percorsi ---")
    paths = generate_paths(S0, r, sigma, T, M, N, device)
    
    # Cashflows iniziali a scadenza (Put Payoff)
    S_T = paths[:, -1]
    cashflows = torch.maximum(K - S_T, torch.tensor(0.0, device=device))
    
    discount_factor = np.exp(-r * dt)
    
    print("--- Inizio Backward Induction (Turbo Mode) ---")
    start_time = time.time()
    
    # Loop all'indietro
    for t in range(M - 1, 0, -1):
        # 1. Sconto cashflow attuali
        cashflows = cashflows * discount_factor
        S_t = paths[:, t]
        
        # 2. Selezione percorsi ITM
        itm_mask = S_t < K # Put Option
        num_itm = torch.sum(itm_mask).item()
        
        # Se abbiamo pochi dati, saltiamo il training per evitare instabilità
        if num_itm < 100:
            continue
            
        # 3. Preparazione Dati Training
        # Usiamo solo i percorsi ITM per la regressione
        X_itm = S_t[itm_mask].reshape(-1, 1)
        Y_itm = cashflows[itm_mask].reshape(-1, 1) # Questi sono i cashflow futuri realizzati
        
        # Normalizzazione: Semplice scaling rispetto allo Strike
        X_norm = X_itm / K
        
        # 4. Training Rete (Reset ad ogni step)
        model = ContinuationNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Training loop più aggressivo (100 epoche)
        model.train()
        for _ in range(100): 
            pred = model(X_norm)
            loss = loss_fn(pred, Y_itm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 5. Decisione (Esercizio Anticipato?)
        model.eval()
        with torch.no_grad():
            continuation_value = model(X_norm).flatten()
            exercise_value = K - X_itm.flatten()
            
            # Logica: Esercito se Valore Immediato > Valore Continuazione Stimato
            should_exercise = exercise_value > continuation_value
            
            # 6. Aggiornamento Cashflow
            # Trucco per aggiornare solo gli indici corretti nel tensore originale grande
            # Creiamo un vettore di indici globali che corrispondono alla maschera ITM
            global_indices = torch.nonzero(itm_mask).flatten()
            
            # Di questi, prendiamo quelli dove abbiamo deciso di esercitare
            exercise_indices = global_indices[should_exercise]
            
            # Aggiorniamo i cashflow: chi esercita prende (K - S_t), chi no tiene il vecchio cashflow
            cashflows[exercise_indices] = exercise_value[should_exercise]

        # Barra di avanzamento rudimentale
        if t % 10 == 0:
            print(f"Step {t}/{M} - ITM Paths: {num_itm}")

    # Step finale: sconto a t=0
    cashflows = cashflows * discount_factor
    price = torch.mean(cashflows).item()
    
    elapsed = time.time() - start_time
    print(f"Tempo calcolo: {elapsed:.2f}s")
    return price

if __name__ == "__main__":
    price = run_deep_lsm_turbo()
    
    # Benchmark teorico (Longstaff-Schwartz Table 1)
    true_price = 4.478
    
    print(f"\n--- RISULTATI TURBO (N={N}) ---")
    print(f"Deep LSM Price:   {price:.4f}")
    print(f"Benchmark Price:  {true_price:.4f}")
    diff = abs(price - true_price)
    print(f"Differenza:       {diff:.4f}")
    print(f"Errore Relativo:  {diff/true_price*100:.2f}%")