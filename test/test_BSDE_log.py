import sys
import os
import time
import numpy as np 
import matplotlib.pyplot as plt 

#--PATH--
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from lsm.bsde.general_solver import GeneralBSDESolver

# Soluzioni analitiche (per confronto)
def analitycal_solution_log(t, X, T, r, sigma):
    """
    Calcola la soluzione analitica per il problema
    Y_T = ln(X_t)
    """
    tau = T - t #maturity
    df = np.exp(-r*tau) #fattore di sconto

    #calcolo Y(t)
    #E[ln(X_T)] = ln(X_t) + (r - 0.5 * sigma^2) * tau
    expected_log_X = np.log(X) + (r - 0.5 * sigma**2) * tau
    y_exact = df * expected_log_X

    #calcolo Z(t)
    #Z = sigma * X * delta --> delta = dY/dX
    #dY/dX = df * (1/X)
    #Z = sigma * X * df * (1/X) = sigma * df --> Z e' costante rispetto a X
    z_exact = np.full_like(X, sigma * df)

    return y_exact, z_exact

#Funzione Terminale G(X_T)
def terminal_function(X_T):
    """
    Funzione terminale G(X_T) = ln(X_T)
    """
    return np.log(np.maximum(X_T, 1e-10)) #evito logaritmi di zero

# --- SALVATAGGIO CSV ---
def save_results_csv(output_dir, params, y_true, y_sim, err_rel, duration):
    """Salva i risultati numerici in un CSV."""
    import csv
    from datetime import datetime
    
    filename = os.path.join(output_dir, "bsde_log_results.csv")
    file_exists = os.path.isfile(filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        "Timestamp": timestamp,
        "Y_True": f"{y_true:.6f}",
        "Y_Sim": f"{y_sim:.6f}",
        "Error_Rel_Pct": f"{err_rel:.4e}",
        "Duration_Sec": f"{duration:.4f}",
        **params # Aggiunge parametri
    }
    # Rimuoviamo la funzione dal dizionario parametri se presente
    if 'terminal_function' in data:
        del data['terminal_function']
    if 'terminal_function' in params:
        del params['terminal_function']

    
    fieldnames = ["Timestamp", "Y_Sim", "Y_True", "Error_Rel_Pct", "Duration_Sec"] + list(params.keys())
    
    with open(filename, mode='a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    
    print(f"Risultati salvati in: {filename}")

#test
def test_log_bsde():
    parameters = {
        "X0": 100.0,
        "r": 0.05,
        "sigma": 0.3,
        "T": 1.0,
        "n_paths": 10000,
        "n_steps": 100,
        "basis_knots": 10
    }

    #calcolo il benchmark
    y_true_0, _ = analitycal_solution_log(0, parameters["X0"], parameters["T"], parameters["r"], parameters["sigma"])
    print(f"\n[BENCHMARK] Valore Esatto Y_0: {y_true_0:.4f}")

    #inizializzo il solver
    solver = GeneralBSDESolver(
        S0 = parameters["X0"],
        r = parameters["r"],
        sigma = parameters["sigma"],
        T = parameters["T"],
        n_paths = parameters["n_paths"],
        n_steps = parameters["n_steps"],
        terminal_function = terminal_function,
        basis_knots = parameters["basis_knots"]
    )

    #eseguo la simulazione
    start_time = time.time()
    y_sim, z_matrix, x_matrix = solver.solve()
    end_time = time.time()

    #risultati
    err_rel = abs(y_sim - y_true_0) / abs(y_true_0)
    print(f"\n[SIMULAZIONE] Valore Stimato Y_0: {y_sim:.4f}")
    print(f"[SIMULAZIONE] Errore Relativo: {err_rel:.4e}")
    print(f"[SIMULAZIONE] Tempo di Esecuzione: {end_time - start_time:.4f} secondi")

    # Salvataggio CSV
    # Setup Output
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "result", "bsde_log")
    os.makedirs(output_dir, exist_ok=True)
    
    save_results_csv(output_dir, parameters, y_true_0, y_sim, err_rel, end_time - start_time)

    #analisi grafica
    #analizzo sempre a t= 0.5
    step_idx = parameters["n_steps"] // 2
    t_val = step_idx * (parameters["T"] / parameters["n_steps"])

    #estraggo i valori a t=0.5
    X_sim = x_matrix[:, step_idx]
    Z_sim = z_matrix[:, step_idx]
    
    #curva teorica
    X_grid = np.linspace(X_sim.min(), X_sim.max(), 200)
    #soluzione analitica a t=0.5
    y_true_t, z_true_t = analitycal_solution_log(t_val, X_grid, parameters["T"], parameters["r"], parameters["sigma"])

    plt.figure(figsize=(14, 6))

    # Plot 1: Accuratezza di Z_t (TEST CRUCIALE)
    plt.subplot(1, 2, 1)
    plt.scatter(X_sim, Z_sim, color='teal', alpha=0.1, s=5, label='Z simulato (Spline)')
    plt.plot(X_grid, z_true_t, 'r-', linewidth=3, label='Z teorico (Costante)')
    
    plt.title(f"Controllo Z_t al tempo t={t_val:.1f}\n(Dovrebbe essere una linea piatta!)")
    plt.xlabel("Sottostante X_t")
    plt.ylabel("Valore Hedging Z_t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Funzione Valore Y_t (Verifica Concavità)
    # Estraiamo anche i valori Y simulati internamente non li abbiamo salvati nel return del solver base
    # Ma possiamo vedere se Y_0 è corretto. Qui plotto la forma teorica.
    plt.subplot(1, 2, 2)
    plt.plot(X_grid, y_true_t, 'b-', linewidth=2, label='Y teorico (Logaritmico)')
    plt.title(f"Forma della Funzione Valore Y(t={t_val:.1f})")
    plt.xlabel("X_t")
    plt.ylabel("Y_t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Salvataggio Plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"bsde_log_plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Grafico salvato in: {plot_filename}")
    plt.close()

if __name__ == "__main__":
    test_log_bsde()