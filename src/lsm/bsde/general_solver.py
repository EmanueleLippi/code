from numpy import sign
import numpy as np 
from typing import Callable, Optional, Tuple
from lsm.basis.spline_basis import SplineBasis

class GeneralBSDESolver:
    """
    Classe generica per risolvere un'equzione BSDE del tipo:
    dY_t = Z_t dW_t
    Y_T = G(X_T)

    Permettendo di passare come input la funzione G(X_T)
    """

    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        n_paths: int,
        n_steps: int,
        terminal_function: Callable[[np.ndarray], np.ndarray],
        basis_knots: int = 10
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_paths = n_paths 
        self.n_steps = n_steps 
        self.g = terminal_function #Funzione terminale G(X_T) dello user
        self.basis_knots = basis_knots

        #parametri derivati
        self.dt = T / n_steps 
        self.df = np.exp(-r * self.dt) #fattore di sconto per 1 step

    #genero i path di X
    def _generate_paths(self) -> np.ndarray:
        """
        Simulazione Forward del processo X
        """
        Z = np.random.normal(0,1,(self.n_paths, self.n_steps))
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        vol = self.sigma * np.sqrt(self.dt)

        log_return = drift + vol * Z
        log_paths = np.cumsum(log_return, axis=1)

        #stato iniziale X_0
        zeros = np.zeros((self.n_paths, 1))
        log_paths = np.hstack([zeros, log_paths])

        # X = X0 * exp(log_paths)
        X = self.S0 * np.exp(log_paths)
        return X

    def solve(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Esegue la backward induction per risolvere la BSDE
        Ritorna:
        Y0: valore di Y al tempo t=0
        Z_matrix: matrice dei controlli Z (n_paths x n_steps)
        S: matrice dei percorsi sottostanti (n_paths x n_steps) --> utile per grafici
        """

        #simulo in avanti
        X = self._generate_paths()

        #imposto la condizione terminale
        Y = self.g(X[:,-1])

        #matrice Z
        Z = np.zeros((self.n_paths, self.n_steps))

        #backward induction
        for i in range(self.n_steps - 1, 0, -1):
            X_current = X[:, i]
            Y_discounted = Y * self.df

            #regressione con la Spline
            try:
                spline = SplineBasis(n_knots=self.basis_knots, degree=3)
                spline.fit(X_current, Y_discounted)

                #aggiorno Y
                Y = spline.evaluate(X_current)

                #calcolo Z
                dY_dX = spline.evaluate_derivative(X_current, order = 1)
                Z[:,i] = self.sigma * X_current * dY_dX

            except Exception as e:
                # Fallback: se fit fallisce, teniamo i valori scontati (martingale assumption)
                # questo evita che Y rimanga fermo al passo t+1
                print(f"[WARNING] Fit Fallito allo step {i}: {e}. Usando fallback valutativo.")
                Y = Y_discounted
        
        #calcolo finale Y0
        Y0 = np.mean(Y * self.df)
        return Y0, Z, X
