import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --- CONFIGURAZIONE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Risaliamo fino a 'src' per vedere i package
src_path = os.path.join(current_dir, '..', 'src') 
sys.path.append(src_path)

# Importiamo la tua classe originale
# Assicurati che il percorso dell'import corrisponda alla tua struttura cartelle
try:
    from lsm.longstaff_schwarz.lsm_spline_engine import LSMSplineEngine
    from lsm.basis.spline_basis import SplineBasis
except ImportError:
    # Fallback se la struttura è leggermente diversa (es. senza 'lsm.')
    from longstaff_schwarz.lsm_spline_engine import LSMSplineEngine
    from basis.spline_basis import SplineBasis

# --- CLASSE PER LA VISUALIZZAZIONE ---
class VisualLSMEngine(LSMSplineEngine):
    """
    Estende il motore originale per catturare i dati interni
    senza modificare il codice sorgente originale.
    """
    def run_with_debug(self):
        # 1. Genera percorsi (uguale all'originale)
        S = self._generate_paths()
        V = self._payoff(S[:, -1])

        # Variabili per salvare i dati di debug
        debug_data = {}
        target_step = self.n_steps // 2  # Fotografiamo l'algoritmo a metà strada

        # Backward induction
        for i in range(self.n_steps - 1, 0, -1):
            S_current = S[:, i]
            V_discounted = V * self.df
            h = self._payoff(S_current)
            itm_idx = np.where(h > 0)[0]

            if len(itm_idx) > max(self.basis_knots, 10):
                x_train = S_current[itm_idx]
                y_train = V_discounted[itm_idx]

                # Fitting
                spline = SplineBasis(n_knots=self.basis_knots, degree=3)
                spline.fit(x_train, y_train)
                continuation_value = spline.evaluate(x_train)

                exercise_now = h[itm_idx] > continuation_value

                # --- CATTURA DATI (Solo allo step target) ---
                if i == target_step:
                    print(f"\n[DEBUG] Cattura dati allo step temporale {i} (t={i*self.dt:.2f})")
                    debug_data['step'] = i
                    debug_data['S_all'] = S # Tutti i percorsi
                    debug_data['x_itm'] = x_train # Prezzi ITM
                    debug_data['y_raw'] = y_train # Cashflow futuri scontati (rumorosi)
                    debug_data['y_pred'] = continuation_value # Curva Spline
                    debug_data['payoff_immediate'] = h[itm_idx] # Valore intrinseco
                    debug_data['knots'] = spline.knots # I nodi della spline
                    debug_data['exercise_mask'] = exercise_now # Chi esercita?

                # Update standard
                V_new = V_discounted.copy()
                global_exercise_idx = itm_idx[exercise_now]
                V_new[global_exercise_idx] = h[global_exercise_idx]
                V = V_new
            else:
                V = V_discounted

        # Calcolo finale prezzo
        V_t0 = V * self.df
        price = np.mean(V_t0)
        std_error = np.std(V_t0) / np.sqrt(self.n_paths)

        return price, std_error, debug_data

# --- FUNZIONI DI PLOTTING E SALVATAGGIO ---
def save_results(params, price, std_error, duration, output_dir):
    """Salva i risultati numerici in un CSV."""
    import csv
    from datetime import datetime
    
    filename = os.path.join(output_dir, "results.csv")
    file_exists = os.path.isfile(filename)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Intestazioni fissi + parametri + risultati
    fieldnames = ["Timestamp", "Price", "StdError", "Duration_Sec"] + list(params.keys())
    
    with open(filename, mode='a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        row = {
            "Timestamp": timestamp,
            "Price": f"{price:.6f}",
            "StdError": f"{std_error:.6f}",
            "Duration_Sec": f"{duration:.4f}"
        }
        # Aggiungi i parametri alla riga
        row.update(params)
        writer.writerow(row)
    
    print(f"Risultati salvati in: {filename}")

def plot_lsm_analysis(params, price, debug_data, output_dir):
    """Genera una dashboard grafica completa e la salva."""
    
    # Estrazione dati
    S_all = debug_data['S_all']
    x_itm = debug_data['x_itm']
    y_raw = debug_data['y_raw']
    y_pred = debug_data['y_pred']
    payoff = debug_data['payoff_immediate']
    knots = debug_data['knots']
    mask_ex = debug_data['exercise_mask']
    step_idx = debug_data['step']
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])
    fig.suptitle(f"Analisi Longstaff-Schwartz (LSM) con B-Spline\nPrezzo Stimato: {price:.4f}", fontsize=16)

    # --- GRAFICO 1: Simulazione Monte Carlo ---
    ax1 = plt.subplot(gs[0, :])
    steps_arr = np.arange(S_all.shape[1])
    ax1.plot(steps_arr, S_all[:100, :].T, color='gray', alpha=0.3, linewidth=0.5)
    
    ax1.axvline(step_idx, color='red', linestyle='--', label=f'Step Analizzato ({step_idx})')
    ax1.axhline(params['K'], color='blue', linestyle=':', label='Strike Price K')
    
    ax1.set_title("Evoluzione Percorsi Asset (Primi 100)", fontsize=12)
    ax1.set_ylabel("Prezzo Asset S(t)")
    ax1.set_xlabel("Time Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- GRAFICO 2: Regressione Spline ---
    ax2 = plt.subplot(gs[1, 0])
    ax2.scatter(x_itm, y_raw, c='gray', alpha=0.4, s=15, label='Cashflow Futuri (Scontati)')
    
    sort_idx = np.argsort(x_itm)
    ax2.plot(x_itm[sort_idx], y_pred[sort_idx], 'r-', linewidth=3, label='Valore Continuazione (B-Spline)')
    
    unique_knots = np.unique(knots)
    for k in unique_knots:
        if min(x_itm) <= k <= max(x_itm):
            ax2.axvline(k, color='green', linestyle=':', alpha=0.6, linewidth=1)
    ax2.plot([], [], 'g:', label='Nodi (Quantili)')

    ax2.set_title(f"Regressione B-Spline allo Step {step_idx}", fontsize=12)
    ax2.set_xlabel("Prezzo Asset Corrente S(t)")
    ax2.set_ylabel("Valore Atteso Futuro")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- GRAFICO 3: Esercizio ---
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(x_itm[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='Valore Continuazione (Hold)')
    ax3.plot(x_itm[sort_idx], payoff[sort_idx], 'b--', linewidth=2, label='Valore Esercizio (Immediate)')
    
    exercise_x = x_itm[mask_ex]
    exercise_y = payoff[mask_ex]
    ax3.scatter(exercise_x, exercise_y, color='orange', s=10, alpha=0.6, label='Punti Esercizio Ottimale', zorder=5)

    ax3.set_title("Frontiera di Esercizio Ottimale", fontsize=12)
    ax3.set_xlabel("Prezzo Asset Corrente S(t)")
    ax3.set_ylabel("Valore")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Salvataggio
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"plot_lsm_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Grafico salvato in: {plot_filename}")
    plt.close() # Chiude la figura per liberare memoria

# --- MAIN TEST FUNCTION ---
def test_pricing_visual():
    print("-- TEST VISUALE: LSM con Grafici Dettagliati --")
    
    # 0. Setup Cartelle
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "result", "longstaff_schwarz", "bspline")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Cartella output: {output_dir}")

    # Parametri
    params = {
        "S0": 36.0,
        "K": 40.0,
        "r": 0.06,
        "T": 1.0,
        "sigma": 0.2,
        "n_steps": 100,
        "n_paths": 10000,
        "basis_knots" : 10,
        "is_call": False
    }

    start_time = time.time()
    
    # Esecuzione
    engine = VisualLSMEngine(**params)
    price, std_error, debug_data = engine.run_with_debug()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- RISULTATI ---")
    print(f"Prezzo: {price:.4f}")
    print(f"Errore standard: {std_error:.4f}")
    print(f"Tempo calcolo: {duration:.4f} s")
    
    # Salvataggio Dati e Grafici
    print("Salvataggio risultati in corso...")
    save_results(params, price, std_error, duration, output_dir)
    plot_lsm_analysis(params, price, debug_data, output_dir)

if __name__ == "__main__":
    test_pricing_visual()