import torch
import matplotlib.pyplot as plt
import numpy as np # Serve per i grafici finali
from NN_torch import PricingModel
from gbm_torch import generate_path

# 0. Setup Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Dispositivo in uso: {device}")

# 1. Parametri e Dati
S0, r, sigma, T, M, N = 100.0, 0.05, 0.2, 1.0, 10, 5000  # 5000 Scenari
K = 100.0

# Generiamo i dati
paths = generate_path(S0, r, sigma, T, M, N)
S_T = paths[:, -1].reshape(-1, 1)

# --- FASE DI NORMALIZZAZIONE (Pre-processing) ---
mu = S_T.mean()
sigma_stat = S_T.std() # Rinomino per non confondere con la sigma del GBM

# Normalizziamo l'input per la rete (Z-score)
S_T_normalized = (S_T - mu) / sigma_stat 

# I target (payoffs)
payoffs = torch.maximum(S_T - K, torch.tensor(0.0, device=device))

# 2. Inizializzazione Modello e Ottimizzatore
model_Price = PricingModel(input_size=1, hidden_size=64, output_size=1).to(device)
optimizer = torch.optim.Adam(model_Price.parameters(), lr=0.01)

# --- NUOVO: SCHEDULER ---
# Ogni 500 epoche, moltiplica il Learning Rate per 0.5 (dimezza)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# 3. TRAINING LOOP
print(f"Media dati: {mu:.2f}, Std dati: {sigma_stat:.2f}")
print("Inizio addestramento...")

loss_history = []
lr_history = []

epochs = 2000
for epoch in range(epochs):
    # Forward pass con dati normalizzati
    prediction = model_Price(S_T_normalized) 
    
    loss = torch.nn.functional.mse_loss(prediction, payoffs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # --- NUOVO: Tracking e Step dello Scheduler ---
    # 1. Salviamo il LR corrente (prima che lo scheduler lo cambi)
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    loss_history.append(loss.item())
    
    # 2. Aggiorniamo il Learning Rate per la prossima epoca
    #scheduler.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

# --- 4. VISUALIZZAZIONE LEARNING RATE & LOSS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Grafico Loss (Sinistra)
ax1.plot(loss_history, label='MSE Loss', color='blue')
ax1.set_title("Andamento dell'Errore (Loss)")
ax1.set_xlabel("Epoca")
ax1.set_ylabel("MSE Loss")
ax1.set_yscale('log') # Scala logaritmica per vedere meglio i miglioramenti
ax1.grid(True, alpha=0.3)
ax1.legend()

# Grafico Learning Rate (Destra)
ax2.plot(lr_history, label='Learning Rate', color='orange')
ax2.set_title("Andamento del Learning Rate (Step Decay)")
ax2.set_xlabel("Epoca")
ax2.set_ylabel("LR Value")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.show()

# --- 5. TEST DI PRICING FINALE ---
# Usiamo i grafici per verificare la qualità della previsione
with torch.no_grad():
    # Creiamo un range di test
    S_test_range = torch.linspace(50, 150, 100).reshape(-1, 1).to(device)
    
    # IMPORTANTE: Normalizziamo i dati di test con la STESSA mu e sigma del training
    S_test_norm = (S_test_range - mu) / sigma_stat
    
    # Calcolo predizioni e conversione per grafico
    pred_payoff = model_Price(S_test_norm).cpu().numpy().flatten()
    S_test_cpu = S_test_range.cpu().numpy().flatten()
    true_payoff = [max(s - K, 0) for s in S_test_cpu]

# Grafico Pricing
plt.figure(figsize=(10, 6))
plt.plot(S_test_cpu, true_payoff, 'k--', label='Payoff Teorico', linewidth=2)
plt.plot(S_test_cpu, pred_payoff, 'r-', label='Predizione Rete Neurale', linewidth=2)
plt.title(f"Pricing Finale (Loss: {loss.item():.5f})")
plt.xlabel("Prezzo Sottostante $S_T$")
plt.ylabel("Payoff")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Test Puntuale su S=110
test_val_raw = torch.tensor([[110.0]], device=device)
test_val_norm = (test_val_raw - mu) / sigma_stat
pred = model_Price(test_val_norm).item()
print(f"\nTEST PUNTUALE:")
print(f"Prezzo S=110 -> Normalizzato={test_val_norm.item():.3f} -> Pred={pred:.2f} (Target reale: 10.00)")