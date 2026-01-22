import sys
import os 
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from lsm.longstaff_schwarz.lsm_spline_engine import LSMSplineEngine

def test_pricing():
    print("-- TEST: LSM --")
    params = {
        "S0": 36.0,
        "K": 40.0,
        "r": 0.06,
        "T": 1.0,
        "sigma": 0.2,
        "n_steps": 100,
        "n_paths": 10000,
        "basis_knots" : 12,
        "is_call": False
    }

    start_time = time.time()
    engine = LSMSplineEngine(**params)
    price, std_error = engine.run()
    end_time = time.time()
    print(f"Prezzo: {price:.4f}")
    print(f"Errore standard: {std_error:.4f}")
    print(f"Tempo di esecuzione: {end_time - start_time:.4f} secondi")

if __name__ == "__main__":
    test_pricing()