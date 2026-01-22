import numpy as np
# Se usi la struttura a cartelle separate (basis e longstaff_schwarz allo stesso livello):
from lsm.basis.spline_basis import SplineBasis
# Se usi la struttura nidificata, lascia il tuo import originale.
from typing import Tuple

class LSMSplineEngine:
    """
    Implementazione dell'algoritmo di Longstaff-Schwarz (LSM)
    utilizzando B-spline come base di regressione.
    """
    def __init__(self, S0: float, K: float, r: float, T: float, sigma: float, 
                 n_steps: int, n_paths: int, basis_knots: int = 8, is_call: bool = False):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.basis_knots = basis_knots
        self.is_call = is_call
        self.dt = T / n_steps
        self.df = np.exp(-r * self.dt)
        
    def _generate_paths(self) -> np.ndarray:
        # 1. Generazione numeri casuali
        Z = np.random.normal(0, 1, (self.n_paths, self.n_steps))
        
        # 2. Parametri GBM
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        vol = self.sigma * np.sqrt(self.dt)
        
        # 3. Calcolo log-rendimenti
        log_returns = drift + vol * Z
        
        # 4. Somma cumulativa (percorso logaritmico)
        log_paths = np.cumsum(log_returns, axis=1)
        
        # 5. Aggiunta stato iniziale t=0
        zeros = np.zeros((self.n_paths, 1))
        log_paths_with_start = np.hstack([zeros, log_paths])
        
        # 6. Conversione in prezzi
        S = self.S0 * np.exp(log_paths_with_start)
        return S
    
    def _payoff(self, S: np.ndarray) -> np.ndarray:
        return np.maximum(S - self.K, 0.0) if self.is_call else np.maximum(self.K - S, 0.0)
    
    def run(self) -> Tuple[float, float]:
        S = self._generate_paths()
        
        # Inizializzazione al tempo T (scadenza)
        V = self._payoff(S[:, -1])

        # Backward induction: da T-1 fino a 1
        for i in range(self.n_steps - 1, 0, -1):
            S_current = S[:, i]
            V_discounted = V * self.df  # Portiamo indietro il valore di continuazione

            h = self._payoff(S_current) # Valore intrinseco (se esercito ora)

            itm_idx = np.where(h > 0)[0]

            # --- CHECK ---
            # Per fittare una Spline servono abbastanza punti. 
            # Se abbiamo meno punti dei nodi o del grado (3), la regressione esplode.
            if len(itm_idx) > max(self.basis_knots, 10):
                x_train = S_current[itm_idx]
                y_train = V_discounted[itm_idx]

                try:
                    spline = SplineBasis(n_knots=self.basis_knots, degree=3)
                    spline.fit(x_train, y_train)
                    continuation_value = spline.evaluate(x_train)

                    exercise_now = h[itm_idx] > continuation_value
                    
                    # Logica di update:
                    # 1. Copiamo i valori scontati (per chi NON esercita)
                    V_new = V_discounted.copy()
                    
                    # 2. Sovrascriviamo SOLO chi esercita
                    global_exercise_idx = itm_idx[exercise_now]
                    V_new[global_exercise_idx] = h[global_exercise_idx]
                    
                    V = V_new

                except Exception:
                    # Fallback: se la regressione fallisce (es. matrice singolare),
                    # assumiamo di non esercitare (o usiamo solo valore scontato)
                    V = V_discounted
            else:
                # Troppi pochi percorsi ITM per fare una stima statistica affidabile:
                # non esercitiamo anticipatamente.
                V = V_discounted

        # Step finale: sconto da t=1 a t=0
        V_t0 = V * self.df
        price = np.mean(V_t0)
        std_error = np.std(V_t0) / np.sqrt(self.n_paths)
            
        return price, std_error