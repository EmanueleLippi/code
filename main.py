import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE PATH ---
# Aggiunge la cartella 'src' al path per importare i moduli interni
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

# --- IMPORT MODULI ---
from lsm.bsde.general_solver import GeneralBSDESolver
from lsm.modelli.bs import bs_price, bs_delta

# ==============================================================================
# 1. DEFINIZIONE DELLE FUNZIONI ANALITICHE E TERMINALI
# ==============================================================================

def get_model_config(model_name, S0, K, T, r, sigma):
    """
    Restituisce la funzione terminale G(x) e le funzioni analitiche 
    per il confronto in base al modello scelto.
    """
    model_name = model_name.lower()

    # --- CASO A: CALL OPTION ---
    if model_name == 'call':
        # G(x) = max(S - K, 0)
        terminal_fn = lambda x: np.maximum(x - K, 0.0)
        
        # Soluzione Esatta (Black-Scholes)
        def exact_sol(t, S_grid):
            tau = T - t
            y = bs_price(S_grid, K, tau, r, sigma, 'call')
            z = sigma * S_grid * bs_delta(S_grid, K, tau, r, sigma, 'call')
            return y, z
            
        return terminal_fn, exact_sol, "Call Option (Convex)"

    # --- CASO B: LOG CONTRACT ---
    elif model_name == 'log':
        # G(x) = ln(x)
        terminal_fn = lambda x: np.log(np.maximum(x, 1e-10))
        
        # Soluzione Esatta
        def exact_sol(t, S_grid):
            tau = T - t
            df = np.exp(-r * tau)
            # Y = df * (ln(S) + (r - 0.5*sigma^2)*tau)
            y = df * (np.log(S_grid) + (r - 0.5 * sigma**2) * tau)
            # Z = sigma * df (Costante)
            z = np.full_like(S_grid, sigma * df)
            return y, z
            
        return terminal_fn, exact_sol, "Log Contract (Concave)"

    # --- CASO C: QUADRATIC ---
    elif model_name == 'quad':
        # G(x) = x^2
        terminal_fn = lambda x: x**2
        
        # Soluzione Esatta
        def exact_sol(t, S_grid):
            tau = T - t
            factor = np.exp((r + sigma**2) * tau)
            y = (S_grid**2) * factor
            z = 2 * sigma * y
            return y, z
            
        return terminal_fn, exact_sol, "Quadratic Y=X^2 (Strong Convex)"

    else:
        raise ValueError(f"Modello '{model_name}' non riconosciuto. Usa: call, log, quad")

# ==============================================================================
# 2. FUNZIONE DI PLOTTING
# ==============================================================================

def plot_results(model_label, t_val, S_sim, Z_sim, S_grid, y_curve, z_curve, Y_sim_val=None):
    plt.figure(figsize=(14, 6))

    # Grafico Sinistro: Controllo Z_t (Hedging)
    plt.subplot(1, 2, 1)
    plt.scatter(S_sim, Z_sim, color='teal', alpha=0.15, s=5, label='Z Simulato (Spline)')
    plt.plot(S_grid, z_curve, 'r-', linewidth=2.5, label='Z Teorico')
    plt.title(f"{model_label}\nControllo Z_t (Hedging) al tempo t={t_val:.2f}")
    plt.xlabel("Sottostante S_t")
    plt.ylabel("Z_t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Grafico Destro: Funzione Valore Y_t
    plt.subplot(1, 2, 2)
    plt.plot(S_grid, y_curve, 'b-', linewidth=2, label='Y Teorico')
    # Nota: Non plottiamo Y simulato puntuale qui perché il solver restituisce solo Y_0 
    # e le matrici S e Z, ma non salva l'intera matrice Y per risparmiare memoria.
    # Possiamo però indicare Y_0 nel titolo.
    title = f"Forma Teorica Valore Y(t={t_val:.2f})"
    if Y_sim_val:
        title += f"\nPrezzo a t=0 stimato: {Y_sim_val:.4f}"
    plt.title(title)
    plt.xlabel("Sottostante S_t")
    plt.ylabel("Y_t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================

def main():
    # --- Configurazione Argomenti da Riga di Comando ---
    parser = argparse.ArgumentParser(description="BSDE Solver con B-Spline Regression")
    
    parser.add_argument('--model', type=str, default='call', choices=['call', 'log', 'quad'],
                        help="Scegli il modello: 'call', 'log', 'quad'")
    
    parser.add_argument('--S0', type=float, default=100.0, help="Prezzo Iniziale")
    parser.add_argument('--K', type=float, default=100.0, help="Strike (usato solo per Call)")
    parser.add_argument('--r', type=float, default=0.05, help="Risk-free rate")
    parser.add_argument('--sigma', type=float, default=0.2, help="Volatilità")
    parser.add_argument('--T', type=float, default=1.0, help="Maturity")
    
    parser.add_argument('--paths', type=int, default=10000, help="Numero percorsi MC")
    parser.add_argument('--steps', type=int, default=50, help="Step temporali")
    parser.add_argument('--knots', type=int, default=10, help="Numero nodi B-Spline")
    
    args = parser.parse_args()

    # --- Setup ---
    print(f"\n=== BSDE SOLVER: {args.model.upper()} ===")
    print(f"Params: S0={args.S0}, r={args.r}, sigma={args.sigma}, T={args.T}")
    print(f"Simulazione: {args.paths} paths, {args.steps} steps, {args.knots} knots")

    # Recupera le funzioni specifiche del modello
    try:
        terminal_fn, exact_sol_fn, label = get_model_config(
            args.model, args.S0, args.K, args.T, args.r, args.sigma
        )
    except ValueError as e:
        print(f"Errore: {e}")
        return

    # --- Calcolo Benchmark (Y_0 esatto) ---
    y_true_0, _ = exact_sol_fn(0, args.S0) # t=0
    print(f"\n[BENCHMARK] Valore Teorico Y_0: {y_true_0:.4f}")

    # --- Esecuzione Solver ---
    solver = GeneralBSDESolver(
        S0=args.S0, r=args.r, sigma=args.sigma, T=args.T,
        n_paths=args.paths, n_steps=args.steps,
        terminal_function=terminal_fn,
        basis_knots=args.knots
    )

    start = time.time()
    y_sim, z_matrix, s_matrix = solver.solve()
    end = time.time()

    # --- Risultati ---
    err_rel = abs(y_sim - y_true_0) / abs(y_true_0) if y_true_0 != 0 else 0
    print(f"[SOLVER]    Valore Simulato Y_0: {y_sim:.4f}")
    print(f"[METRICHE]  Errore Relativo:     {err_rel*100:.3f}%")
    print(f"            Tempo di calcolo:    {end - start:.2f} s")

    # --- Generazione Grafici ---
    # Analizziamo un punto intermedio (t=0.5 circa)
    step_idx = args.steps // 2
    t_val = step_idx * (args.T / args.steps)
    
    # Dati simulati a quel tempo
    S_slice = s_matrix[:, step_idx]
    Z_slice = z_matrix[:, step_idx]
    
    # Dati teorici per le curve
    S_grid = np.linspace(min(S_slice), max(S_slice), 200)
    y_curve, z_curve = exact_sol_fn(t_val, S_grid)
    
    plot_results(label, t_val, S_slice, Z_slice, S_grid, y_curve, z_curve, y_sim)

if __name__ == "__main__":
    main()