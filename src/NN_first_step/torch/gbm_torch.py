import torch
import math

device = torch.device("mps" if torch.mps.is_available() else "cpu")

def generate_path(S0, r, sigma, T, M, N):
    dt = T/M
    S = torch.zeros((N, M+1), device=device)
    S[:, 0] = S0
    Z = torch.randn((N, M), device=device)
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    log_return = torch.cumsum(drift + vol * Z, dim=1)
    S[:, 1:] = S0 * torch.exp(log_return)
    return S