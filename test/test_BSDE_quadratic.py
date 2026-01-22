from IPython.core.pylabtools import figsize
import sys
import os
import time
import numpy as np 
import matplotlib.pyplot as plt 

# --- CONFIGURAZIONE PATH ---

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir,'..', 'src')
sys.path.append(src_path)

from lsm.bsde.solver import LinearBSDESolver
from lsm.basis.spline_basis import SplineBasis

# Soluzioni teoriche con formule analitiche
def analytic_solution(t, S, T, r, sigma):
    """
    Calcola Y e Z esatti per il problema
    Y_T = S_T^2
    """

    tau = T - t 
    #formula derivativa: E[e^-rT * S_T^2 | S_T]
    #Y(t) = e^-r*tau * E[S_T^2]
    #E[S_T^2] = S^2 * e^(2r + sigma^2)*tau
    #Y(t) = S^2 * e^(r + sigma^2)*tau
    factor = np.exp((r + sigma**2) * tau)
    y_exact = S**2 * factor

    #Z(t) = sigma * S * dY/dS
    #dY/dS = 2 * S * factor
    #Z = sigma * S * (2 * S * factor) = 2 * sigma * Y
    z_exact = 2 * sigma * y_exact
    
    return y_exact, z_exact
    
# -- Definisco un solver per il problema --> extend della classe LinearBSDESolver per il problema quadratico --
class QuadraticBSDESolver(LinearBSDESolver):
    """
    Sovrascrivo il metodo _g per il problema quadratico
    """
    def _g(self, x):
        return x**2 #condizione terminale quadratica
    
    # Metodo per estrarre i dati per i grafici
    def run_debug(self):
        S = self._generate_paths()
        Y = self._g(S[:,-1])
        Z_matrix = np.zeros((self.n_paths, self.n_steps))

        for i in range(self.n_steps - 1, 0, -1):
            S_curr = S[:, i]
            Y_discounted = Y * self.df

            spline = SplineBasis(n_knots=self.basis_knots, degree=3)
            spline.fit(S_curr, Y_discounted)

            Y = spline.evaluate(S_curr)

            #calcolo Z
            delta = spline.evaluate_derivative(S_curr, order=1)
            Z_t = delta * self.sigma * S_curr
            Z_matrix[:, i] = Z_t
        Y_t0 = np.mean(Y * self.df)
        return Y_t0, Z_matrix, S

# --- SALVATAGGIO ---
def save_results_csv(output_dir, params, y_true, y_sim, err_rel, duration):
    """Salva i risultati numerici in un CSV."""
    import csv
    from datetime import datetime
    
    filename = os.path.join(output_dir, "bsde_quadratic_results.csv")
    file_exists = os.path.isfile(filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        "Timestamp": timestamp,
        "Y_True": f"{y_true:.6f}",
        "Y_Sim": f"{y_sim:.6f}",
        "Error_Rel_Pct": f"{err_rel*100:.4f}",
        "Duration_Sec": f"{duration:.4f}",
        **params
    }
    
    fieldnames = ["Timestamp", "Y_Sim", "Y_True", "Error_Rel_Pct", "Duration_Sec"] + list(params.keys())
    
    with open(filename, mode='a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    
    print(f"Risultati salvati in: {filename}")

# TEST
def test_quadratic_bsde():
    print("--- TEST BSDE SOLVER: QUADRATICO ---")
    
    # Setup Output
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "result", "bsde_quadratic")
    os.makedirs(output_dir, exist_ok=True)
    
    # Parametri Ottimizzati
    params = {
        "S0": 10.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "N_paths": 20000,    # Aumentato da 5000: riduce varianza Monte Carlo
        "N_steps": 50,
        "Knots": 20          # Aumentato da 12: cattura meglio la curvatura x^2
    }

    #calcolo soluzione teorica
    y_true_0, _ = analytic_solution(0, params["S0"], params["T"], params["r"], params["sigma"])
    print(f"[BENCHMARK] Y_0 esatto: {y_true_0:.4f}")

    #creo il solver e eseguo il calcolo
    solver = QuadraticBSDESolver(
        params["S0"], 0, params["r"], params["sigma"], params["T"], 
        params["N_paths"], params["N_steps"], basis_knots=params["Knots"]
    )
    
    #tempo di calcolo
    start_time = time.time()
    y_sim, z_matrix, s_matrix = solver.run_debug()
    end_time = time.time()
    duration = end_time - start_time

    err_rel = abs(y_sim - y_true_0) / y_true_0

    print(f"[SOLVER]    Y_0 (BSDE Spline): {y_sim:.4f}")
    print(f"[METRICHE]  Errore Relativo: {err_rel*100:.2f}%")
    print(f"            Tempo calcolo:   {duration:.2f} s")
    
    # Salvataggio CSV
    save_results_csv(output_dir, params, y_true_0, y_sim, err_rel, duration)

    # ANALISI GRAFICA
    #prendiamo un istante t a metà per analisi grafica
    step_idx = params["N_steps"] // 2
    t_val = step_idx * (params["T"]/params["N_steps"])

    #dati simulati
    S_t_sim = s_matrix[:, step_idx]
    Z_t_sim = z_matrix[:, step_idx]
    
    #curva teorica
    S_grid = np.linspace(min(S_t_sim), max(S_t_sim), 200)
    _, Z_curve = analytic_solution(t_val, S_grid, params["T"], params["r"], params["sigma"])

    plt.figure(figsize=(12,6))

    #grafico 1: accuratezza di Z_t
    plt.subplot(1,2,1)
    plt.scatter(S_t_sim, Z_t_sim, alpha=0.2, s=5, label='BSDE Z_t (Simulato)')
    plt.plot(S_grid, Z_curve, 'r-', linewidth=2, label='Z teorico')
    plt.title(f"Controllo Z_t al tempo t={t_val:.2f}") #da aspettarsi sia una parabola
    plt.xlabel("S_t")
    plt.ylabel("Z_t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    #grafico 2: istogramma dei residui
    plt.subplot(1,2,2)
    _, Z_exact_points = analytic_solution(t_val, S_t_sim, params["T"], params["r"], params["sigma"])
    residuals = Z_t_sim - Z_exact_points

    plt.hist(residuals, bins=50, alpha=0.6, color='blue', edgecolor='black', density=True)
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Distribuzione dei residui su Z_t")
    plt.xlabel("Residui (Z_t - Z_t esatto)")
    plt.ylabel("Densità")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Salvataggio Plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"bsde_quadratic_plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Grafico salvato in: {plot_filename}")
    plt.close()

if __name__ == "__main__":
    test_quadratic_bsde()
    
    
