import numpy as np
from scipy.interpolate import BSpline
from numpy.linalg import lstsq
from typing import Optional, List


class SplineBasis:
    """
    B-spline basis for regression (used in Longstaff–Schwartz / LSM).
    """

    def __init__(self, n_knots: int = 8, degree: int = 3):
        self.n_knots = n_knots
        self.degree = degree
        self.knots: Optional[np.ndarray] = None
        self.basis: Optional[List[BSpline]] = None
        self.beta: Optional[np.ndarray] = None

    def _build_knots(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).ravel()
        knots_internal = np.quantile(x, np.linspace(0, 1, self.n_knots))

        # (opzionale) elimina duplicati
        knots_internal = np.unique(knots_internal)
        if len(knots_internal) < 2:
            raise ValueError("Not enough distinct knot locations from data.")

        knots = np.concatenate((
            np.repeat(knots_internal[0], self.degree),
            knots_internal,
            np.repeat(knots_internal[-1], self.degree)
        ))
        return knots

    def _build_basis(self, knots: np.ndarray) -> List[BSpline]:
        n_basis = len(knots) - self.degree - 1
        basis: List[BSpline] = []

        for i in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0
            basis.append(BSpline(knots, coeffs, self.degree))

        return basis

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        self.knots = self._build_knots(x)
        self.basis = self._build_basis(self.knots)

        design_matrix = np.column_stack([b(x) for b in self.basis])
        self.beta, _, _, _ = lstsq(design_matrix, y, rcond=None)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.beta is None or self.basis is None:
            raise RuntimeError("Spline has not been fitted")

        x = np.asarray(x)
        x_1d = np.atleast_1d(x).ravel()

        # Strategia Ottimizzata: 
        # Usa direttamente l'oggetto BSpline di SciPy che è calcolato in C ed è molto più veloce
        # rispetto alla costruzione manuale della design matrix [b(x) for b in basis].
        full_spline = BSpline(self.knots, self.beta, self.degree)
        out = full_spline(x_1d)

        # se input era scalare, torna scalare
        return out[0] if x.ndim == 0 else out.reshape(x.shape)

    def evaluate_derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        if self.beta is None or self.knots is None or self.basis is None:
            raise RuntimeError("Spline has not been fitted")
        
        x = np.asarray(x)
        x_1d = np.atleast_1d(x).ravel()

        # Strategia Ottimizzata:
        # Invece di sommare le derivate delle singole basi, costruiamo
        # un singolo oggetto BSpline che rappresenta l'intera curva fittata.
        # Scipy gestisce i nodi e i coefficienti internamente.

        #self.beta contiene i pesi corretti per le basi B-Spline standard
        #quindi possiamo creare un oggetto BSpline con i coefficienti corretti
        #e valutarne la derivata
        
        full_spline = BSpline(self.knots, self.beta, self.degree)
        
        #calcolo della derivata --> return di una BSpline nuova
        deriv_spline = full_spline.derivative(nu = order)

        #valutazione della derivata in x_1d
        out = deriv_spline(x_1d)

        #se input era scalare, torna scalare
        return out[0] if x.ndim == 0 else out.reshape(x.shape)