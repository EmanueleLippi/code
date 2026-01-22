import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../src")
sys.path.append(src_path)
try:
    from lsm.basis.spline_basis import SplineBasis
except ImportError as e:
    print("ERRORE DI IMPORTAZIONE:")
    print(f"Non riesco a trovare il modulo 'lsm'. Percorso aggiunto: {src_path}")
    print("Assicurati di aver creato i file __init__.py nelle cartelle lsm e basis.")
    raise e

def test_spline():
    """
    Simula un moto browniano e testa l'adattamento di una spline
    """
    np.random.seed(42)

    # parametri della simulazione
    N_paths = 2000
    S0 = 100.0
    VOL = 0.2
    T = 1.0
    MU = 0.0

    print(f"--- START TEST: BSPLINE FITTING ---")

    # generazione del moto browniano
    brownian_motion = np.random.normal(0,1,N_paths)
    X_data = S0 * np.exp((MU - 0.5 * VOL**2) * T + VOL * np.sqrt(T) * brownian_motion)

    #funzione target (Y_true) e Rumore (Y_noisy)
    # il target e' una parabola
    def true_function(s):
        return 10 + (s - 100)**2 / 50

    Y_true = true_function(X_data)

    noise_level = 5.0
    Y_noisy = Y_true + np.random.normal(0, noise_level, N_paths)

    #Istanzio la Base Spline dalla mia classe
    print("--- INIZIALIZZAZIONE SPLINE ... ---")
    spline = SplineBasis(n_knots=8, degree=3)

    print("--- FITTING DEI DATI ... ---")
    spline.fit(X_data, Y_noisy)
    
    print(f"--- NODI GENERATI (QUANTILI): {len(spline.knots)}")

    # Valuation
    X_plot = np.linspace(np.min(X_data), np.max(X_data), 500)
    Y_plot_pred = spline.evaluate(X_plot)
    Y_plot_true = true_function(X_plot)

    # Metriche
    Y_pred_on_data = spline.evaluate(X_data)
    mse = mean_squared_error(Y_true, Y_pred_on_data)
    r2 = r2_score(Y_true, Y_pred_on_data)
    
    print(f"\n--- RISULTATI ---")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Prepare output directory
    output_dir = os.path.join(current_dir, "../result/spline_test")
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["MSE", mse])
        writer.writerow(["R2", r2])
    print(f"Metrics saved to {csv_path}")

    # 6. Plotting
    plot_path = os.path.join(output_dir, "test_spline.png")
    plot_results(X_data, Y_noisy, X_plot, Y_plot_true, Y_plot_pred, spline.knots, mse, plot_path)

def plot_results(X_data, Y_noisy, X_plot, Y_true, Y_pred, knots, mse, save_path):
    plt.figure(figsize=(10, 6))
    
    # Scatter dati rumorosi
    plt.scatter(X_data, Y_noisy, alpha=0.15, color='gray', s=10, label='Simulazioni (Noise)')
    
    # Linee
    plt.plot(X_plot, Y_true, 'k--', linewidth=2, label='Funzione Vera')
    plt.plot(X_plot, Y_pred, 'r-', linewidth=3, label='B-Spline Approssimata')
    
    # Nodi
    unique_knots = np.unique(knots)
    for k in unique_knots:
        plt.axvline(x=k, color='green', linestyle=':', alpha=0.5)

    plt.title(f"Test Modulo SplineBasis (MSE: {mse:.2f})")
    plt.xlabel("Underlying Price S_T")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show() # Commented out to avoid blocking execution if running non-interactively
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    test_spline()