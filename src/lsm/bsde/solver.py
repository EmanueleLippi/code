import numpy as np
from lsm.basis.spline_basis import SplineBasis

class LinearBSDESolver:
    """
    classe per risolvere l'equazione BSDE utilizzando il metodo Longstaff-Schwartz con B-Spline
    sistema di riferimento
    dY_t = Z_t dW_t
    Y_T = g(X_T)
    
    """
    def __init__(self, S0, K, r, sigma, T, n_paths, n_steps, basis_knots = 10):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.basis_knots = basis_knots
        self.dt = T / n_steps
        self.df = np.exp(-r * self.dt) # fattore di sconto per 1 step
    
    def _generate_paths(self):
        """
        Genera percorsi di asset utilizzando il modello di Geometric Brownian Motion
        """
        Z = np.random.normal(0, 1, (self.n_paths, self.n_steps))

        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        vol = self.sigma * np.sqrt(self.dt)

        log_return = drift + vol * Z
        log_paths = np.cumsum(log_return, axis=1)
        
        zeros = np.zeros((self.n_paths, 1))
        log_paths = np.hstack([zeros, log_paths])
        
        S = self.S0 * np.exp(log_paths)
        return S
    
    def _g(self, x):
        """
        condizione terminale del sistema G(X_T)
        """
        return np.maximum(x - self.K, 0)
        
    def run(self):
        """
        Risolve la BSDE dell'equazione per trovare Y_0
        """
        # 1. Forward: Genero i percorsi di X
        X = self._generate_paths()

        # 2. Condizione di termine Y_T = g(X_T)
        Y = self._g(X[:,-1])

        # Matrice degli Z --> Z[i,t] e' il valore di Z al tempo t per il percorso i
        Z = np.zeros((self.n_paths, self.n_steps))

        # 3. Backward Induction da T-1 a 0
        for i in range(self.n_steps - 1, 0, -1):
            X_current = X[:, i]

            #sconto il valore di Y dal tempo t_{i+1} al tempo t_i
            Y_discounted = Y * self.df #rappresentano i valori grezzi (rumorosi) osservati

            #regressione
            #stimiamo Y(t_i) = E[Y_discounted | X_current]
            #utilizzo la spline per trovare la funzione che lega X_current a Y_discounted
            spline = SplineBasis(n_knots=self.basis_knots, degree=3)
            spline.fit(X_current, Y_discounted)

            #Aggiorno, sostituendo i valori rumorosi con quelli stimati dalla spline
            #fondamentale nelle BSDE per stabilizzare la varianza quando
            #andremo a calcolare Z_t
            Y = spline.evaluate(X_current)

            #Calcolo di Z_t --> Z_t = dY/dX * sigma * X_t

            #calcoliamo la derivata prima della curva
            delta = spline.evaluate_derivative(X_current, order=1)
            #applico la formula
            Z[:, i] = delta * self.sigma * X_current

        #4. Calcolo finale da t_1 --> t_0
        # in t_0 abbiamo X e' costante (X_0) quindi basta una semplice media
        Y_t0 = np.mean(Y*self.df)

        # Anche a t=0 possiamo calcolare Z_0 (sarà un numero unico, non un vettore)
        # Rifare il fit su una spline veloce su X_0 (che è costante) non ha senso,
        # ma teoricamente Z_0 è il Delta dell'opzione a t=0 * sigma * X_0.
        # Possiamo approssimarlo con la media dei Z al primo step (t=1) o lasciarlo a 0.
        return Y_t0, Z

    