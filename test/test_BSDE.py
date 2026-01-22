import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE PATH ---
# Aggiungiamo la cartella 'src' al path per importare i moduli
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

# --- IMPORT MODULI PROGETTO ---
from lsm.bsde.solver import LinearBSDESolver
from lsm.basis.spline_basis import SplineBasis
# NUOVO: Importiamo le formule analitiche dal modulo modelli
from lsm.modelli.bs import bs_price, bs_delta

# --- WRAPPER PER DEBUG ---
class VisualBSDESolver(LinearBSDESolver):
    """
    Estende il solver standard per restituire anche la matrice dei percorsi S.
    Necessario perché il solver base restituisce solo (Price, Z_matrix),
    ma per fare i grafici di Z vs S abbiamo bisogno anche di S.
    """
    def run_debug(self):
        # 1. Genera percorsi (metodo della classe base)
        S = self._generate_paths()
        
        # 2. Inizializzazione condizione terminale
        Y = self._g(S[:, -1])
        
        # Matrice per salvare Z
        Z_matrix = np.zeros((self.n_paths, self.n_steps))

        # 3. Backward Loop
        for i in range(self.n_steps - 1, 0, -1):
            S_curr = S[:, i]
            Y_discounted = Y * self.df 

            # Regressione
            spline = SplineBasis(n_knots=self.basis_knots, degree=3)
            spline.fit(S_curr, Y_discounted)
            
            # Update Y (Proiezione)
            Y = spline.evaluate(S_curr)
            
            # Calcolo Z (Hedging/Controllo)
            # Z_t = dY/dX * sigma * X_t
            delta = spline.evaluate_derivative(S_curr, order=1)
            Z_t = delta * self.sigma * S_curr
            Z_matrix[:, i] = Z_t

        # Passo finale a t=0
        Y_t0 = np.mean(Y * self.df)
        
        # Restituiamo Y_0, la matrice Z e ANCHE la matrice S per i grafici
        return Y_t0, Z_matrix, S

# --- FUNZIONI UTILI ---
def save_results_csv(output_dir, params, bs_val, y_bsde, err_abs, err_rel, duration):
    """Salva i risultati numerici in un CSV."""
    import csv
    from datetime import datetime
    
    filename = os.path.join(output_dir, "bsde_results.csv")
    file_exists = os.path.isfile(filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Uniamo tutti i dati in un dizionario
    data = {
        "Timestamp": timestamp,
        "Benchmark_BS": f"{bs_val:.6f}",
        "Price_BSDE": f"{y_bsde:.6f}",
        "Error_Abs": f"{err_abs:.6f}",
        "Error_Rel_Pct": f"{err_rel*100:.4f}",
        "Duration_Sec": f"{duration:.4f}",
        **params # Aggiunge S0, K, T, etc.
    }
    
    fieldnames = ["Timestamp", "Price_BSDE", "Benchmark_BS", "Error_Abs", "Error_Rel_Pct", "Duration_Sec"] + list(params.keys())
    
    with open(filename, mode='a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    
    print(f"Risultati numerici salvati in: {filename}")

# --- FUNZIONE DI TEST PRINCIPALE ---
def test_bsde_accuracy():
    print("--- TEST BSDE SOLVER: ACCURATEZZA Y e Z ---")
    
    # 0. Setup Cartelle Output
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "result", "bsde")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Cartella output: {output_dir}")

    # 1. Parametri della simulazione
    params = {
        'S0': 100.0,
        'K': 100.0,
        'T': 1.0,
        'r': 0.05,
        'sigma': 0.2,
        'n_paths': 5000,
        'n_steps': 50,
        'basis_knots': 12
    }

    # 2. Calcolo Valore Teorico (Benchmark)
    bs_val = bs_price(params['S0'], params['K'], params['T'], 
                      params['r'], params['sigma'], option_type='call')
    
    print(f"\n[BENCHMARK] Call Europea (Black-Scholes): {bs_val:.4f}")

    # 3. Esecuzione del Solver BSDE
    solver = VisualBSDESolver(**params)
    
    start_time = time.time()
    y_bsde, z_matrix, s_matrix = solver.run_debug()
    end_time = time.time()
    duration = end_time - start_time

    # 4. Metriche di errore sul Prezzo (Y_0)
    err_abs = y_bsde - bs_val
    err_rel = abs(err_abs) / bs_val
    
    print(f"[SOLVER]    Call Europea (BSDE Spline):   {y_bsde:.4f}")
    print(f"[METRICHE]  Errore Relativo: {err_rel*100:.2f}%")
    print(f"            Tempo calcolo:   {duration:.2f} s")

    # 5. Salvataggio Risultati Numerici
    save_results_csv(output_dir, params, bs_val, y_bsde, err_abs, err_rel, duration)

    # --- ANALISI GRAFICA DI Z_t ---
    step_idx = params['n_steps'] // 2
    time_val = step_idx * (params['T'] / params['n_steps'])
    time_to_maturity = params['T'] - time_val

    # Dati simulati
    S_t_sim = s_matrix[:, step_idx]
    Z_t_sim = z_matrix[:, step_idx]

    # Curva Teorica
    S_grid = np.linspace(min(S_t_sim), max(S_t_sim), 200)
    # Calcoliamo delta esatto
    delta_exact = bs_delta(S_grid, params['K'], time_to_maturity, 
                           params['r'], params['sigma'], option_type='call')
    Z_exact = params['sigma'] * S_grid * delta_exact

    # --- PLOTTING ---
    plt.figure(figsize=(14, 6))

    # Grafico 1: Scatter Plot Z vs S
    plt.subplot(1, 2, 1)
    plt.scatter(S_t_sim, Z_t_sim, alpha=0.15, color='gray', s=5, label='BSDE Z_t (Simulato)')
    plt.plot(S_grid, Z_exact, 'r-', linewidth=2.5, label='Analitico (sigma*S*Delta)')
    
    plt.title(f"Verifica Controllo Z_t al tempo t={time_val:.2f}")
    plt.xlabel(f"Prezzo Sottostante S(t={time_val:.2f})")
    plt.ylabel("Controllo Z(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Grafico 2: Istogramma dei Residui
    delta_sim_points = bs_delta(S_t_sim, params['K'], time_to_maturity, 
                                params['r'], params['sigma'], option_type='call')
    Z_exact_points = params['sigma'] * S_t_sim * delta_sim_points
    residuals = Z_t_sim - Z_exact_points

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, color='royalblue', alpha=0.7, density=True, edgecolor='black', linewidth=0.5)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Distribuzione Errore Stima Z_t (Residui)")
    plt.xlabel("Errore (Z_simulato - Z_teorico)")
    plt.ylabel("Densità")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 6. Salvataggio Grafico
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"bsde_plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Grafico salvato in: {plot_filename}")
    plt.close() # Chiude la figura

if __name__ == "__main__":
    test_bsde_accuracy()