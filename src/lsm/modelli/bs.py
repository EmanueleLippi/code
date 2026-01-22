import numpy as np 
import scipy.stats as stats 

def bs_price(S, K, T, r, sigma, option_type='call'):
    """
    Calcola il prezzo di un'opzione Europea (Call o Put) 
    usando la formula chiusa di Black-Scholes-Merton.
    
    Parameters:
    -----------
    S : float o np.ndarray - Prezzo Spot del sottostante
    K : float - Strike Price
    T : float - Tempo a scadenza (in anni)
    r : float - Tasso privo di rischio (annuale)
    sigma : float - Volatilità
    option_type : str - 'call' o 'put' (case insensitive)
    
    Returns:
    --------
    price : float o np.ndarray
    """
    # Assicuriamo che T non sia esattamente 0 per evitare divisioni per zero
    # Se T è 0, il prezzo è il payoff intrinseco
    T = np.maximum(T, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    
    return price

def bs_delta(S, K, T, r, sigma, option_type='call'):
    """
    Calcola il delta di un'opzione Europea (Call o Put) 
    usando la formula chiusa di Black-Scholes.
    """
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type.lower() == 'call':
        delta = stats.norm.cdf(d1)
    elif option_type.lower() == 'put':
        delta = stats.norm.cdf(d1) - 1.0
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    
    return delta
    