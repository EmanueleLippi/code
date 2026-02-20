from scipy.optimize import minimize
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def is_day(t):
    hour = t % 24
    return (hour >= 7) & (hour < 19)

def design_matrix(H, t, K=1):
    day = is_day(t).astype(float)
    night = 1.0 - day

    X = []

    # -H_t term
    X.append(-H * day)
    X.append(-H * night)

    # constant term
    X.append(day)
    X.append(night)

    for k in range(1, K+1):
        omega = 2 * np.pi * k / 24
        cos = np.cos(omega * t)
        sin = np.sin(omega * t)

        X.append(cos * day)
        X.append(cos * night)
        X.append(sin * day)
        X.append(sin * night)

    return np.column_stack(X)

def mu_t(t, a0, alpha, beta):
    """
    alpha, beta: lists of Fourier coefficients
    """
    mu = a0
    K = len(alpha)
    for k in range(1, K + 1):
        omega = 2 * np.pi * k / 24
        mu += alpha[k-1] * np.cos(omega * t)
        mu += beta[k-1] * np.sin(omega * t)
    return mu
def mu_t_daynight(t, a0_day, a0_night,
                  alpha_day, alpha_night,
                  beta_day, beta_night):
    mu_day = mu_t(t, a0_day, alpha_day, beta_day)
    mu_night = mu_t(t, a0_night, alpha_night, beta_night)
    return np.where(is_day(t), mu_day, mu_night)

def kappa_t(t, kappa_day, kappa_night):
    return np.where(is_day(t), kappa_day, kappa_night)

def sigma_t(t, sigma_day, sigma_night):
    return np.where(is_day(t), sigma_day, sigma_night)

def simulate_ou_day_night(
    H0,
    params,
    T,
    Nsim=1000,
    dt=1.0,
    seed=42
):
    a0_day, a0_night = params["a0_day"], params["a0_night"]
    alpha_day, alpha_night = params["alpha_day"], params["alpha_night"]
    beta_day, beta_night = params["beta_day"], params["beta_night"]
    kappa_day, kappa_night = params["kappa_day"], params["kappa_night"]
    sigma_day, sigma_night = params["sigma_day"], params["sigma_night"]

    np.random.seed(seed)

    Nt = int(T / dt)
    paths = np.zeros((Nsim, Nt + 1))
    paths[:, 0] = H0

    for i in range(Nt):
        t_actual = i * dt # Calculate the actual time in hours
        mu = mu_t_daynight(
    t_actual,
    a0_day, a0_night,
    alpha_day, alpha_night,
    beta_day, beta_night
)

        kappa = kappa_t(t_actual, kappa_day, kappa_night)
        sigma = sigma_t(t_actual, sigma_day, sigma_night)

        Z = np.random.randn(Nsim)

        paths[:, i+1] = (
            paths[:, i]
            + kappa * (mu - paths[:, i]) * dt
            + sigma * Z * np.sqrt(dt) # Corrected diffusion term: sigma * Z * sqrt(dt)
        )

    return paths

def prepare_H(filepath, n=1, mul_factor=1):
    """Given a csv_path returns a np array of the energy deficit with hourly data.
    If the input data is not hourly makes a mean and squashes the data to be hourly.
    n is the number of data points for hour (15min data => n=4)
    If the data is not in the right units it is multiplied by mul_factor
    """
    df = pd.read_csv(filepath)
    H = df["Consumo (W)"] - df["Produzione (W)"]
    H = np.array(H)*mul_factor
    n_hours = len(H) //  n
    H_hourly = H[:n*n_hours].reshape(n_hours, n).mean(axis=1)
    return H_hourly

def prepare_S(filepath,n=1,mul_factor=1):
    """As above but for excel data and prices"""
    df = pd.read_excel(filepath)
    S=np.array(df["€/MWh"])
    S = np.char.replace(S.astype(str),',','.').astype(np.float64) * mul_factor
    n_hours = len(S) //  n
    S_hourly = S[:n*n_hours].reshape(n_hours, n).mean(axis=1)
    return S_hourly

def calibrate_OU_variable(H,K,dt=1):
    #I am not sure that dt != 1 works
    Y = H[1:] - H[:-1]
    t = np.arange(len(Y)) * dt
    X = design_matrix(H[:-1], t, K=K)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    theta = reg.coef_
    kappa_day   = theta[0] / dt
    kappa_night = theta[1] / dt

    a0_day   = theta[2] / kappa_day
    a0_night = theta[3] / kappa_night

    theta_k = theta[4:4 + 4*K].reshape(K, 4)

    alpha_day   = theta_k[:, 0] / kappa_day
    alpha_night = theta_k[:, 1] / kappa_night
    beta_day    = theta_k[:, 2] / kappa_day
    beta_night  = theta_k[:, 3] / kappa_night

    residuals = Y - reg.predict(X)

    day_idx = is_day(t)
    night_idx = ~day_idx

    sigma_day = np.std(residuals[day_idx], ddof=1)
    sigma_night = np.std(residuals[night_idx], ddof=1)
    return {"kappa_day":kappa_day,"kappa_night":kappa_night,
            "a0_day":a0_day,"a0_night":a0_night,
            "alpha_day":alpha_day,"alpha_night":alpha_night,
            "beta_day":beta_day,"beta_night":beta_night,
            "sigma_day":sigma_day,"sigma_night":sigma_night}

def generate_plot(paths,real_path,dt_real,dt_sim,T, name, save=False, save_name="0000"):
    lower = np.percentile(paths,5,axis=0)
    upper = np.percentile(paths,95,axis=0)
    time_sim = np.arange(paths.shape[1])*dt_sim
    time_real = np.arange(real_path.shape[0])*dt_real
    plt.figure(figsize=(12,6))
    plt.plot(time_real[:int(T/dt_real)],real_path[:int(T/dt_real)], label=f"Real {name}", color = "green")
    plt.fill_between(time_sim,lower[:len(time_sim)],upper[:len(time_sim)],alpha=0.3,label="80% Band")
    plt.legend()
    plt.title(f"OU Simulation with periodic mean of {name}")
    plt.xlabel("Hours")
    plt.ylabel(name)
    if save:
        plt.savefig(f"plots/{name}_{save_name}_{T}h.png")
    else:
        plt.show()
