import os
import time
import argparse
import json
import csv
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow.compat.v1 as tf
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    _PLOTTING_AVAILABLE = False

tf.disable_v2_behavior()

from abc import ABC, abstractmethod


###############################################################################
# Classe madre e prima figlia (tenute volutamente molto vicine alla tua base)
###############################################################################


class FBSNN(ABC):  # Forward-Backward Stochastic Neural Network
    def __init__(
        self,
        Xi_generator,
        T,
        M,
        N,
        D,
        layers,
        clip_grad_norm=1.0,
        use_antithetic_sampling=True,
        log_device_placement=False,
    ):
        self.Xi_generator = Xi_generator  # initial data generator
        self.T = T  # terminal time

        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions

        self.layers = layers  # (D+1) --> 1

        self.clip_grad_norm = clip_grad_norm
        self.use_antithetic_sampling = bool(use_antithetic_sampling)
        self.log_device_placement = bool(log_device_placement)

        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=self.log_device_placement,
            )
        )

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[M, self.N + 1, 1])  # M x (N+1) x 1
        self.W_tf = tf.placeholder(tf.float32, shape=[M, self.N + 1, self.D])  # M x (N+1) x D
        self.Xi_tf = tf.placeholder(tf.float32, shape=[M, D])  # M x D
        self.const_tf = tf.placeholder(tf.float32, shape=[])

        self.loss, self.X_pred, self.Y_pred, self.Z_pred = self.loss_function(
            self.t_tf, self.W_tf, self.Xi_tf
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients_and_variables = self.optimizer.compute_gradients(self.loss)
        if self.clip_grad_norm is not None:
            non_null_grads_and_vars = [(g, v) for g, v in gradients_and_variables if g is not None]
            if non_null_grads_and_vars:
                gradients, variables = zip(*non_null_grads_and_vars)
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grad_norm)
                clipped = list(zip(gradients, variables))
                untouched = [(g, v) for g, v in gradients_and_variables if g is None]
                gradients_and_variables = clipped + untouched
        self.train_op = self.optimizer.apply_gradients(gradients_and_variables)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

        # in-memory snapshot utilities for early stopping
        self._trainable_vars = tf.trainable_variables()
        self._snapshot_ph = [
            tf.placeholder(tf.float32, shape=v.shape, name=f"snapshot_ph_{i}")
            for i, v in enumerate(self._trainable_vars)
        ]
        self._restore_ops = [v.assign(ph) for v, ph in zip(self._trainable_vars, self._snapshot_ph)]

    def save_model(self, path):
        save_path = self.saver.save(self.sess, path)
        print(f"Model saved in path: {save_path}")

    def load_model(self, path):
        self.saver.restore(self.sess, path)
        print(f"Model restored from path: {path}")

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t, X):  # M x 1, M x D
        u = self.neural_net(tf.concat([t, X], 1), self.weights, self.biases)  # M x 1
        Du = tf.gradients(u, X)[0]  # M x D
        return u, Du

    def Dg_tf(self, X):  # M x D
        return tf.gradients(self.g_tf(X), X)[0]  # M x D

    def loss_function(self, t, W, Xi):  # M x (N+1) x 1, M x (N+1) x D, M x D
        loss = 0
        X_list = []
        Y_list = []
        Z_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi  # già M x D
        Y0, Du0 = self.net_u(t0, X0)  # M x 1, M x D
        sigma0 = self.sigma_tf(t0, X0, Y0)
        Z0 = tf.squeeze(tf.matmul(tf.expand_dims(Du0, 1), sigma0), axis=1)

        X_list.append(X0)
        Y_list.append(Y0)
        Z_list.append(Z0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            dW = W1 - W0
            sigma_dW = tf.squeeze(tf.matmul(sigma0, tf.expand_dims(dW, -1)), axis=[-1])
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + sigma_dW

            Y1_tilde = (
                Y0
                + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0)
                + tf.reduce_sum(Z0 * dW, axis=1, keepdims=True)
            )

            Y1, Du1 = self.net_u(t1, X1)
            sigma1 = self.sigma_tf(t1, X1, Y1)
            Z1 = tf.squeeze(tf.matmul(tf.expand_dims(Du1, 1), sigma1), axis=1)

            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            sigma0 = sigma1

            X_list.append(X0)
            Y_list.append(Y0)
            Z_list.append(Z0)

        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))

        Dg = self.Dg_tf(X1)
        Z_terminal = tf.squeeze(tf.matmul(tf.expand_dims(Dg, 1), sigma1), axis=1)
        loss += tf.reduce_sum(tf.square(Z1 - Z_terminal))

        X = tf.stack(X_list, axis=1)
        Y = tf.stack(Y_list, axis=1)
        Z = tf.stack(Z_list, axis=1)

        return loss / self.N, X, Y, Z

    def fetch_minibatch(self):
        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1), dtype=np.float32)
        DW = np.zeros((M, N + 1, D), dtype=np.float32)
        dt = self.T / N

        Dt[:, 1:, :] = dt
        if self.use_antithetic_sampling and M > 1:
            half_M = M // 2
            DW_half = np.sqrt(dt) * np.random.normal(size=(half_M, N, D))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M : 2 * half_M, 1:, :] = -DW_half
            if M % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * np.random.normal(size=(N, D))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D
        Xi_batch = self.Xi_generator(M, D).astype(np.float32)
        return t, W, Xi_batch

    def _get_snapshot(self):
        return self.sess.run(self._trainable_vars)

    def _restore_snapshot(self, weights):
        feed = {ph: w for ph, w in zip(self._snapshot_ph, weights)}
        self.sess.run(self._restore_ops, feed_dict=feed)

    def train(
        self,
        N_Iter,
        learning_rate,
        const_value=None,
        eval_every=50,
        val_batches=8,
        early_stopping_metric="loss",
        patience=None,
        min_delta=1e-3,
        restore_best=False,
    ):
        start_time = time.time()
        last_loss = None
        current_const = np.float32(self.const if const_value is None else const_value)

        best_score = np.inf
        best_iter = -1
        best_snapshot = None
        no_improve_iters = 0
        stopped_early = False

        for it in range(N_Iter):
            t_batch, W_batch, Xi_batch = self.fetch_minibatch()
            tf_dict = {
                self.Xi_tf: Xi_batch,
                self.t_tf: t_batch,
                self.W_tf: W_batch,
                self.learning_rate: learning_rate,
                self.const_tf: current_const,
            }
            self.sess.run(self.train_op, tf_dict)

            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value, Y_value, learning_rate_value = self.sess.run(
                    [self.loss, self.Y_pred, self.learning_rate], tf_dict
                )
                last_loss = float(loss_value)
                mean_Y0 = np.mean(Y_value[:, 0, 0])
                print(
                    "It: %d, Loss: %.3e, Mean Y0: %.3f, Time: %.2f, Learning Rate: %.3e"
                    % (it, loss_value, mean_Y0, elapsed, learning_rate_value)
                )
                start_time = time.time()

            if (it % eval_every == 0) or (it == N_Iter - 1):
                eval_stats = self.evaluate(const_value=current_const, n_batches=val_batches)
                if early_stopping_metric == "loss":
                    score = eval_stats["mean_loss"]
                else:
                    raise ValueError(f"Unsupported early_stopping_metric='{early_stopping_metric}'")

                if (best_score - score) > min_delta:
                    best_score = float(score)
                    best_iter = int(it)
                    best_snapshot = self._get_snapshot()
                    no_improve_iters = 0
                else:
                    no_improve_iters += eval_every

                if patience is not None and no_improve_iters >= patience:
                    print(f"[EarlyStop] it={it}, best_it={best_iter}, best_score={best_score:.6e}")
                    stopped_early = True
                    break

        if restore_best and best_snapshot is not None:
            self._restore_snapshot(best_snapshot)
            print(f"[RestoreBest] best_it={best_iter}, best_score={best_score:.6e}")

        return {
            "const": float(current_const),
            "learning_rate": float(learning_rate),
            "n_iter": int(N_Iter),
            "last_loss": last_loss,
            "best_iter": int(best_iter),
            "best_score": float(best_score),
            "stopped_early": bool(stopped_early),
        }

    def evaluate(self, const_value=None, n_batches=5):
        current_const = np.float32(self.const if const_value is None else const_value)
        losses = []
        losses_per_sample = []
        y0s = []

        for _ in range(n_batches):
            t_batch, W_batch, Xi_batch = self.fetch_minibatch()
            tf_dict = {
                self.Xi_tf: Xi_batch,
                self.t_tf: t_batch,
                self.W_tf: W_batch,
                self.const_tf: current_const,
            }
            loss_value, y_value = self.sess.run([self.loss, self.Y_pred], tf_dict)
            loss_value = float(loss_value)
            losses.append(loss_value)
            losses_per_sample.append(loss_value / float(self.M))
            y0s.append(list(y_value[:, 0, 0]))

        return {
            "const": float(current_const),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "mean_loss_per_sample": float(np.mean(losses_per_sample)),
            "std_loss_per_sample": float(np.std(losses_per_sample)),
            "mean_y0": float(np.mean(y0s)),
            "std_y0": float(np.std(y0s)),
            "n_batches": int(n_batches),
        }

    def predict(self, Xi_star, t_star, W_star, const_value=None):
        current_const = np.float32(self.const if const_value is None else const_value)
        tf_dict = {
            self.Xi_tf: Xi_star,
            self.t_tf: t_star,
            self.W_tf: W_star,
            self.const_tf: current_const,
        }

        X_star = self.sess.run(self.X_pred, tf_dict)
        Y_star = self.sess.run(self.Y_pred, tf_dict)
        Z_star = self.sess.run(self.Z_pred, tf_dict)
        return X_star, Y_star, Z_star

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        M = self.M
        D = self.D
        return np.zeros([M, D])

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = self.M
        D = self.D
        return tf.matrix_diag(tf.ones([M, D]))


# backward compatibility
fbsde_NN = FBSNN


class NN_Quadratic_Coupled(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, parameters):
        self.mu1 = parameters["mu1"]
        self.mu2 = parameters["mu2"]
        self.c1 = parameters["c1"]
        self.c2 = parameters["c2"]
        self.c3 = parameters["c3"]
        self.c4 = parameters["c4"]
        self.gamma = parameters["gamma"]
        self.s1 = parameters["s1"]
        self.s2 = parameters["s2"]
        self.s3 = parameters["s3"]
        self.x_max = parameters["x_max"]
        self.v_min = parameters["v_min"]
        self.v_max = parameters["v_max"]
        self.d = parameters["d"]
        self.const = parameters["const"]
        super().__init__(Xi, T, M, N, D, layers)

    def psi(self, X_state):
        result = tf.maximum(
            0.0,
            tf.minimum(
                1.0,
                tf.minimum(X_state / self.d, (self.x_max - X_state) / self.d),
            ),
        )
        return result

    def psi3(self, V):
        result = tf.maximum(0.0, tf.minimum(1.0, (self.v_max - V) / self.d))
        return result

    def psi4(self, V):
        result = tf.maximum(0.0, tf.minimum(1.0, (V - self.v_min) / self.d))
        return result

    def f(self, X, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        Z_S, Z_H, Z_V, _ = tf.split(Z, num_or_size_splits=4, axis=1)
        s1 = tf.cast(self.s1, tf.float32)
        gamma = tf.cast(self.gamma, tf.float32)
        exp_S = tf.exp(-S)
        return -0.5 * V * self.psi(-exp_S * Z_S / (gamma * s1))

    def mu_tf(self, t, X, Y, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        mu1 = tf.cast(self.mu1, tf.float32)
        mu2 = tf.cast(self.mu2, tf.float32)
        c1 = tf.cast(self.c1, tf.float32)
        c2 = tf.cast(self.c2, tf.float32)
        c3 = tf.cast(self.c3, tf.float32)
        c4 = tf.cast(self.c4, tf.float32)
        x_max = tf.cast(self.x_max, tf.float32)
        const = tf.cast(self.const_tf, tf.float32)
        dS = mu1 * (c1 - S)
        dH = mu2 * (c2 - H)
        dV = (
            self.f(X, Z) * self.psi(X_state)
            + c3 * self.psi(-X_state) * self.psi3(V)
            - c4 * self.psi(X_state - x_max) * self.psi4(V)
        )
        dX = V
        return tf.concat([dS, dH, dV, dX], axis=1)

    def g_tf(self, X):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        gamma = tf.cast(self.gamma, tf.float32)
        exp_S = tf.exp(S)
        return -gamma * exp_S * X_state + V ** 2 + V * X_state

    def phi_tf(self, t, X, Y, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        Z_S, Z_H, Z_V, _ = tf.split(Z, num_or_size_splits=4, axis=1)

        mu1 = tf.cast(self.mu1, tf.float32)
        c1 = tf.cast(self.c1, tf.float32)
        s1 = tf.cast(self.s1, tf.float32)
        s3 = tf.cast(self.s3, tf.float32)
        c3 = tf.cast(self.c3, tf.float32)
        c4 = tf.cast(self.c4, tf.float32)
        x_max = tf.cast(self.x_max, tf.float32)
        gamma = tf.cast(self.gamma, tf.float32)
        const = tf.cast(self.const_tf, tf.float32)

        exp_S = tf.exp(S)

        term1 = -gamma * exp_S * X_state * mu1 * (c1 - S)
        term2 = (2 * V + X_state) * (
            self.f(X, Z) * self.psi(X_state)
            + c3 * self.psi(-X_state) * self.psi3(V)
            - c4 * self.psi(X_state - x_max) * self.psi4(V)
        )
        term3 = -gamma * exp_S * V + (0.5 * (Z_V / s3 - X_state)) ** 2
        term4 = -0.5 * gamma * exp_S * X_state * s1 ** 2 + s3 ** 2

        return term1 + term2 + term3 + term4

    def sigma_tf(self, t, X, Y):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        s1 = tf.cast(self.s1, tf.float32)
        s2 = tf.cast(self.s2, tf.float32)
        s3 = tf.cast(self.s3, tf.float32)

        zeros = tf.zeros_like(S)
        ones = tf.ones_like(S)

        r1 = tf.concat([s1 * ones, zeros, zeros, zeros], axis=1)
        r2 = tf.concat([zeros, s2 * ones, zeros, zeros], axis=1)
        r3 = tf.concat([zeros, zeros, s3 * ones, zeros], axis=1)
        r4 = tf.concat([zeros, zeros, zeros, zeros], axis=1)

        return tf.stack([r1, r2, r3, r4], axis=1)


###############################################################################
# Estensione ricorsiva (senza alterare la logica delle classi sopra)
###############################################################################


def _as_blob_dict(blob_or_path: Union[Dict[str, np.ndarray], str, None]) -> Optional[Dict[str, np.ndarray]]:
    if blob_or_path is None:
        return None
    if isinstance(blob_or_path, dict):
        return blob_or_path
    if isinstance(blob_or_path, str):
        with np.load(blob_or_path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    raise TypeError("blob_or_path must be dict, str path, or None")


def save_blob_npz(blob: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **blob)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2)


def save_rows_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rows is None or len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    keys = []
    keys_set = set()
    for row in rows:
        for k in row.keys():
            if k not in keys_set:
                keys_set.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_serializable(row.get(k, None)) for k in keys})


def export_standard_parameter_blob(model: FBSNN) -> Dict[str, np.ndarray]:
    values = model.sess.run(model.weights + model.biases)
    n_layers = len(model.weights)
    blob = {
        "n_layers": np.array(n_layers, dtype=np.int32),
        "layers": np.asarray(model.layers, dtype=np.int32),
    }
    for i in range(n_layers):
        blob[f"W_{i}"] = values[i].astype(np.float32)
        blob[f"b_{i}"] = values[n_layers + i].astype(np.float32)
    return blob


def plot_stage_logs(stage_logs: List[Dict], out_prefix: str, title: str) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_stage_logs")
        return
    if stage_logs is None or len(stage_logs) == 0:
        return

    x = np.arange(len(stage_logs))
    loss = np.array([row.get("eval_mean_loss", np.nan) for row in stage_logs], dtype=np.float64)
    y0 = np.array([row.get("eval_mean_y0", np.nan) for row in stage_logs], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    plt.plot(x, loss, "b-o", markersize=3, linewidth=1.2, label="eval mean loss")
    plt.yscale("log")
    plt.title(f"{title} - Eval Mean Loss")
    plt.xlabel("Stage index")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_eval_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y0, "g-o", markersize=3, linewidth=1.2, label="eval mean y0")
    plt.title(f"{title} - Eval Mean Y0")
    plt.xlabel("Stage index")
    plt.ylabel("Mean Y0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_eval_y0.png", dpi=160)
    plt.close()


def _pass_index(pass_id: int) -> int:
    pid = int(pass_id)
    return pid - 1 if pid >= 1 else pid


def _pass_label(pass_id: int) -> str:
    return f"pass{_pass_index(pass_id)}"


def _pass_tag(pass_id: int, width: int = 2) -> str:
    return f"pass{_pass_index(pass_id):0{int(width)}d}"


def plot_recursive_pass_logs_multi(pass_logs_by_pass: Dict[int, List[Dict]], out_dir: str) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_pass_logs_multi")
        return
    if pass_logs_by_pass is None or len(pass_logs_by_pass) == 0:
        return

    normalized = {}
    for pass_id, rows in pass_logs_by_pass.items():
        rows_sorted = sorted(rows or [], key=lambda r: r["block"])
        if len(rows_sorted) > 0:
            normalized[int(pass_id)] = rows_sorted
    if len(normalized) == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    pass_ids = sorted(normalized.keys())
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(pass_ids), 2)))
    use_per_sample_loss = all(
        len(rows) > 0 and ("eval_mean_loss_per_sample" in rows[0]) for rows in normalized.values()
    )
    loss_key = "eval_mean_loss_per_sample" if use_per_sample_loss else "eval_mean_loss"

    plt.figure(figsize=(10, 6))
    for i, pass_id in enumerate(pass_ids):
        rows = normalized[pass_id]
        b = np.array([r["block"] for r in rows], dtype=np.int32)
        l = np.array([r[loss_key] for r in rows], dtype=np.float64)
        plt.plot(
            b,
            l,
            marker="o",
            linewidth=1.5,
            color=colors[i],
            label=f"{_pass_label(pass_id)} loss",
        )
    plt.yscale("log")
    if use_per_sample_loss:
        plt.title("Recursive blocks - Eval Mean Loss per Sample")
        plt.ylabel("Loss / M (log scale)")
    else:
        plt.title("Recursive blocks - Eval Mean Loss")
        plt.ylabel("Loss (log scale)")
    plt.xlabel("Block index")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_blocks_eval_loss.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, pass_id in enumerate(pass_ids):
        rows = normalized[pass_id]
        b = np.array([r["block"] for r in rows], dtype=np.int32)
        y = np.array([r["eval_mean_y0"] for r in rows], dtype=np.float64)
        plt.plot(
            b,
            y,
            marker="o",
            linewidth=1.5,
            color=colors[i],
            label=f"{_pass_label(pass_id)} y0",
        )
    plt.title("Recursive blocks - Eval Mean Y0")
    plt.xlabel("Block index")
    plt.ylabel("Mean Y0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_blocks_eval_y0.png"), dpi=160)
    plt.close()


def plot_recursive_pass_logs(pass1_logs: List[Dict], pass2_logs: List[Dict], out_dir: str) -> None:
    plot_recursive_pass_logs_multi(
        pass_logs_by_pass={
            1: pass1_logs or [],
            2: pass2_logs or [],
        },
        out_dir=out_dir,
    )


def score_pass_logs(
    rows: List[Dict],
    loss_key: str = "eval_mean_loss_per_sample",
    worst_block_weight: float = 0.35,
) -> float:
    losses = np.array([float(r.get(loss_key, np.nan)) for r in (rows or [])], dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        return float("inf")
    return float(np.mean(losses) + worst_block_weight * np.max(losses))


def build_stitched_rollout_inputs(
    blocks: List[Dict[str, float]],
    M: int,
    N_per_block: int,
    D: int,
    seed: int = 1234,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    rollout_inputs = []
    for block in blocks:
        dt = float(block["T_block"]) / float(N_per_block)
        Dt = np.zeros((M, N_per_block + 1, 1), dtype=np.float32)
        DW = np.zeros((M, N_per_block + 1, D), dtype=np.float32)
        Dt[:, 1:, :] = dt

        if M > 1:
            half_M = M // 2
            DW_half = np.sqrt(dt) * rng.normal(size=(half_M, N_per_block, D))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M : 2 * half_M, 1:, :] = -DW_half
            if M % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * rng.normal(size=(N_per_block, D))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * rng.normal(size=(M, N_per_block, D))

        t_abs = float(block["t_start"]) + np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        rollout_inputs.append((t_abs.astype(np.float32), W.astype(np.float32)))
    return rollout_inputs

def print_recursive_pass(
    pass_entries: List[Dict[str, Any]],
    blocks: List[Dict[str, float]],
    rec_dir: str,
    params: Dict[str, np.ndarray],
    N_per_block: int,
    D: int,
    layers: List[int],
    T_total: float,
    exact_solution: Optional[Dict[str, Any]],
    selection_metric: str = "auto",
    exact_regression_tolerance: float = 0.20,
    exact_regression_action: str = "warn",
    eval_bundle_path: str = "",
    eval_seed: int = 1234,
    eval_min_paths: int = 64,
    sample_paths: int = 8,
    enforce_exact_regression_guardrail: bool = True,
    print_compact_logs: bool = True,
) -> Dict[str, Any]:
    if pass_entries is None or len(pass_entries) == 0:
        raise RuntimeError("print_recursive_pass called with empty pass_entries")

    pass_entries = sorted(pass_entries, key=lambda x: int(x["pass_id"]))
    os.makedirs(rec_dir, exist_ok=True)
    plots_dir = os.path.join(rec_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    pass_logs_by_pass = {}
    for p in pass_entries:
        pass_id = int(p["pass_id"])
        pass_idx = _pass_index(pass_id)
        logs = p.get("logs", [])
        pass_logs_by_pass[pass_id] = logs

        if print_compact_logs:
            print(f"\n=== Recursive Log {_pass_label(pass_id)} (compact) ===")
            for row in logs:
                norm_msg = ""
                if "eval_mean_loss_per_sample" in row:
                    norm_msg = f", eval_loss/M={row['eval_mean_loss_per_sample']:.3e}"
                print(
                    f"block={row['block']}, t=[{row['t_start']:.1f},{row['t_end']:.1f}], "
                    f"eval_loss={row['eval_mean_loss']:.3e}{norm_msg}, eval_y0={row['eval_mean_y0']:.3f}, "
                    f"target={row['precision_target']}, refine={row['refine_rounds']}"
                )

        save_rows_csv(logs, os.path.join(rec_dir, f"pass_{pass_idx:02d}_logs.csv"))
        if pass_idx == 0:
            save_rows_csv(logs, os.path.join(rec_dir, "pass0_logs.csv"))
        if pass_idx == 1:
            save_rows_csv(logs, os.path.join(rec_dir, "pass1_logs.csv"))

    plot_recursive_pass_logs_multi(pass_logs_by_pass, plots_dir)

    score_key = "eval_mean_loss_per_sample"
    all_rows = [row for rows in pass_logs_by_pass.values() for row in rows]
    if not all(score_key in row for row in all_rows):
        score_key = "eval_mean_loss"
    pass_scores_loss = {
        int(pass_id): score_pass_logs(rows, loss_key=score_key)
        for pass_id, rows in pass_logs_by_pass.items()
        if len(rows) > 0
    }
    if len(pass_scores_loss) == 0:
        raise RuntimeError("No pass logs available for pass selection")
    best_pass_by_loss = int(min(pass_scores_loss, key=pass_scores_loss.get))
    print(
        f"[Selection:loss] metric={score_key}, best={_pass_label(best_pass_by_loss)}, "
        f"score={pass_scores_loss[best_pass_by_loss]:.6e}"
    )

    eval_bundle_path = str(eval_bundle_path or "").strip()
    if eval_bundle_path == "":
        eval_bundle_path = os.path.join(rec_dir, "evaluation_bundle.npz")
    eval_bundle_path = os.path.abspath(os.path.expanduser(eval_bundle_path))

    if os.path.isfile(eval_bundle_path):
        Xi_stitched, rollout_inputs = load_evaluation_bundle(
            path=eval_bundle_path,
            n_blocks_expected=len(blocks),
            N_per_block_expected=N_per_block,
            D_expected=D,
        )
        print(
            f"[EvalBundle] loaded path={eval_bundle_path}, M={Xi_stitched.shape[0]}, "
            f"blocks={len(rollout_inputs)}"
        )
    else:
        Xi_stitched = make_deterministic_xi_default(
            max(1, int(eval_min_paths)),
            D,
            seed=int(eval_seed),
        )
        rollout_inputs = build_stitched_rollout_inputs(
            blocks=blocks,
            M=Xi_stitched.shape[0],
            N_per_block=N_per_block,
            D=D,
            seed=int(eval_seed),
        )
        save_evaluation_bundle(
            path=eval_bundle_path,
            Xi_initial=Xi_stitched,
            rollout_inputs=rollout_inputs,
            blocks=blocks,
        )
        print(
            f"[EvalBundle] created path={eval_bundle_path}, M={Xi_stitched.shape[0]}, "
            f"seed={int(eval_seed)}"
        )

    stitched_by_pass = {}
    exact_summary_by_pass = {}
    exact_bundle_by_pass = {}
    for p in pass_entries:
        pass_id = int(p["pass_id"])
        pass_tag = _pass_tag(pass_id)
        stitched_pred = predict_recursive_stitched(
            block_blobs=p["blobs"],
            blocks=blocks,
            Xi_initial=Xi_stitched,
            params=params,
            N_per_block=N_per_block,
            D=D,
            layers=layers,
            T_total=T_total,
            rollout_inputs=rollout_inputs,
        )
        stitched_by_pass[pass_id] = stitched_pred

        np.savez(
            os.path.join(rec_dir, f"stitched_predictions_{pass_tag}.npz"),
            t=stitched_pred["t"],
            X=stitched_pred["X"],
            Y=stitched_pred["Y"],
            Z=stitched_pred["Z"],
        )
        plot_recursive_stitched_predictions(
            stitched=stitched_pred,
            blocks=blocks,
            out_dir=plots_dir,
            sample_paths=sample_paths,
            file_suffix=f"_{pass_tag}",
        )

        if exact_solution is not None:
            exact_bundle = compute_stitched_exact_bundle(
                stitched=stitched_pred,
                exact_solution=exact_solution,
            )
            exact_summary = exact_bundle["summary"]
            exact_summary_by_pass[pass_id] = exact_summary
            exact_bundle_by_pass[pass_id] = exact_bundle
            print(
                f"[Exact] {_pass_label(pass_id)} "
                f"mean_pred_Y0={exact_summary['mean_pred_y0']:.6f}, "
                f"mean_exact_Y0={exact_summary['mean_exact_y0']:.6f}, "
                f"abs_err_Y0={exact_summary['abs_error_mean_y0']:.6e}, "
                f"mean_abs_err_Y={exact_summary['mean_abs_error_y']:.6e}, "
                f"mean_abs_err_Z={exact_summary['mean_abs_error_z']:.6e}"
            )

            save_json(
                {
                    "summary": exact_summary,
                    "timeseries": exact_bundle["timeseries"],
                },
                os.path.join(rec_dir, f"exact_metrics_{pass_tag}.json"),
            )
            save_exact_error_timeseries_csv(
                exact_bundle["timeseries"],
                os.path.join(rec_dir, f"exact_errors_{pass_tag}.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=stitched_pred,
                Y_exact=exact_bundle["Y_exact"],
                Z_exact=exact_bundle["Z_exact"],
                blocks=blocks,
                out_dir=plots_dir,
                sample_paths=sample_paths,
                file_suffix=f"_{pass_tag}",
            )

    if (
        enforce_exact_regression_guardrail
        and exact_solution is not None
        and len(exact_summary_by_pass) >= 2
        and str(exact_regression_action) != "ignore"
    ):
        tol = float(exact_regression_tolerance)
        if tol > 0.0:
            sorted_pass_ids = sorted(exact_summary_by_pass.keys())
            prev_id = sorted_pass_ids[0]
            prev_val = float(exact_summary_by_pass[prev_id]["mean_abs_error_y"])
            for pass_id in sorted_pass_ids[1:]:
                curr_val = float(exact_summary_by_pass[pass_id]["mean_abs_error_y"])
                if prev_val > 0.0 and curr_val > prev_val * (1.0 + tol):
                    msg = (
                        "[ExactGuardrail] Regression detected on mean_abs_error_y: "
                        f"{_pass_label(prev_id)}={prev_val:.6e} -> {_pass_label(pass_id)}={curr_val:.6e} "
                        f"(+{(curr_val / prev_val - 1.0) * 100.0:.2f}%, tol={tol * 100.0:.2f}%)"
                    )
                    if str(exact_regression_action) == "error":
                        raise RuntimeError(msg)
                    print(msg)
                prev_id = pass_id
                prev_val = curr_val

    selected_pass_id, selected_score_metric, selected_score, selected_score_by_pass = resolve_pass_selection(
        pass_scores_by_loss=pass_scores_loss,
        exact_summary_by_pass=exact_summary_by_pass,
        selection_metric=str(selection_metric),
        loss_metric_label=score_key,
    )
    print(
        f"[Selection:final] metric={selected_score_metric}, best={_pass_label(selected_pass_id)}, "
        f"score={selected_score:.6e}"
    )

    selected_stitched = stitched_by_pass[selected_pass_id]
    selected_exact_bundle = exact_bundle_by_pass.get(selected_pass_id, None)
    np.savez(
        os.path.join(rec_dir, "stitched_predictions_final.npz"),
        t=selected_stitched["t"],
        X=selected_stitched["X"],
        Y=selected_stitched["Y"],
        Z=selected_stitched["Z"],
    )
    plot_recursive_stitched_predictions(
        stitched=selected_stitched,
        blocks=blocks,
        out_dir=plots_dir,
        sample_paths=sample_paths,
        file_suffix="",
    )

    if exact_solution is not None and selected_exact_bundle is not None:
        save_json(
            {
                "summary": selected_exact_bundle["summary"],
                "timeseries": selected_exact_bundle["timeseries"],
            },
            os.path.join(rec_dir, "exact_metrics_final.json"),
        )
        save_exact_error_timeseries_csv(
            selected_exact_bundle["timeseries"],
            os.path.join(rec_dir, "exact_errors_final.csv"),
        )
        plot_recursive_exact_comparison(
            stitched=selected_stitched,
            Y_exact=selected_exact_bundle["Y_exact"],
            Z_exact=selected_exact_bundle["Z_exact"],
            blocks=blocks,
            out_dir=plots_dir,
            sample_paths=sample_paths,
            file_suffix="",
        )

    plot_recursive_stitched_y_convergence(
        stitched_by_pass=stitched_by_pass,
        blocks=blocks,
        out_dir=plots_dir,
        sample_paths=sample_paths,
    )

    return {
        "processed_pass_ids": sorted(pass_logs_by_pass.keys()),
        "processed_pass_indices": sorted(_pass_index(pid) for pid in pass_logs_by_pass.keys()),
        "score_key": score_key,
        "pass_scores_loss": pass_scores_loss,
        "pass_scores_loss_by_index": {
            str(_pass_index(k)): float(v) for k, v in pass_scores_loss.items()
        },
        "selected_pass_id": int(selected_pass_id),
        "selected_pass_index": int(_pass_index(selected_pass_id)),
        "selected_score_metric": selected_score_metric,
        "selected_score": float(selected_score),
        "selected_scores_by_pass": selected_score_by_pass,
        "selected_scores_by_pass_index": {
            str(_pass_index(int(k))): float(v)
            for k, v in selected_score_by_pass.items()
        },
        "exact_summary_by_pass": exact_summary_by_pass,
        "exact_summary_by_pass_index": {
            str(_pass_index(k)): v for k, v in exact_summary_by_pass.items()
        },
        "eval_bundle_path": eval_bundle_path,
        "evaluation_bundle_M": int(Xi_stitched.shape[0]),
    }


def make_deterministic_xi_default(M: int, D: int, seed: int = 1234) -> np.ndarray:
    if int(D) != 4:
        raise ValueError(f"make_deterministic_xi_default currently supports D=4, got D={int(D)}")
    rng = np.random.RandomState(int(seed))
    Xi = np.zeros((int(M), int(D)), dtype=np.float32)
    Xi[:, 0] = rng.normal(1.0, 1.0, int(M))
    Xi[:, 1] = rng.normal(1.0, 1.0, int(M))
    Xi[:, 2] = rng.normal(0.0, 1.0, int(M))
    Xi[:, 3] = rng.uniform(1.0, 9.0, int(M))
    return Xi.astype(np.float32)


def save_evaluation_bundle(
    path: str,
    Xi_initial: np.ndarray,
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
    blocks: List[Dict[str, float]],
) -> None:
    dir_name = os.path.dirname(path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    t_stack = np.stack([pair[0] for pair in rollout_inputs], axis=0).astype(np.float32)
    w_stack = np.stack([pair[1] for pair in rollout_inputs], axis=0).astype(np.float32)
    t_start = np.array([float(b["t_start"]) for b in blocks], dtype=np.float32)
    t_end = np.array([float(b["t_end"]) for b in blocks], dtype=np.float32)
    np.savez(
        path,
        Xi_initial=np.asarray(Xi_initial, dtype=np.float32),
        t_bundle=t_stack,
        W_bundle=w_stack,
        block_t_start=t_start,
        block_t_end=t_end,
    )


def load_evaluation_bundle(
    path: str,
    n_blocks_expected: int,
    N_per_block_expected: int,
    D_expected: int,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as data:
        Xi = np.asarray(data["Xi_initial"], dtype=np.float32)
        t_bundle = np.asarray(data["t_bundle"], dtype=np.float32)
        W_bundle = np.asarray(data["W_bundle"], dtype=np.float32)

    if Xi.ndim != 2 or Xi.shape[1] != int(D_expected):
        raise ValueError(
            f"Invalid Xi_initial shape in evaluation bundle: {Xi.shape}, expected [M, {int(D_expected)}]"
        )
    if t_bundle.ndim != 4 or W_bundle.ndim != 4:
        raise ValueError(
            f"Invalid rollout bundle rank: t={t_bundle.shape}, W={W_bundle.shape}; expected rank-4"
        )
    if t_bundle.shape[0] != int(n_blocks_expected) or W_bundle.shape[0] != int(n_blocks_expected):
        raise ValueError(
            f"Evaluation bundle blocks mismatch: got {t_bundle.shape[0]}, expected {int(n_blocks_expected)}"
        )
    if t_bundle.shape[2] != int(N_per_block_expected) + 1:
        raise ValueError(
            "Evaluation bundle N_per_block mismatch: "
            f"got {t_bundle.shape[2]-1}, expected {int(N_per_block_expected)}"
        )
    if W_bundle.shape[3] != int(D_expected):
        raise ValueError(
            f"Evaluation bundle D mismatch in W: got {W_bundle.shape[3]}, expected {int(D_expected)}"
        )
    if t_bundle.shape[1] != Xi.shape[0] or W_bundle.shape[1] != Xi.shape[0]:
        raise ValueError(
            "Evaluation bundle M mismatch between Xi and rollout tensors: "
            f"Xi={Xi.shape[0]}, t_bundle={t_bundle.shape[1]}, W_bundle={W_bundle.shape[1]}"
        )

    rollout_inputs = []
    for i in range(int(n_blocks_expected)):
        rollout_inputs.append((t_bundle[i], W_bundle[i]))
    return Xi, rollout_inputs


def resolve_pass_selection(
    pass_scores_by_loss: Dict[int, float],
    exact_summary_by_pass: Dict[int, Dict[str, Any]],
    selection_metric: str,
    loss_metric_label: str = "eval_mean_loss_per_sample",
) -> Tuple[int, str, float, Dict[str, float]]:
    if len(pass_scores_by_loss) == 0:
        raise RuntimeError("resolve_pass_selection called with empty pass_scores_by_loss")

    metric = str(selection_metric or "auto").strip().lower()
    selected_by_loss = int(min(pass_scores_by_loss, key=pass_scores_by_loss.get))

    if metric in ("", "auto"):
        if len(exact_summary_by_pass) > 0:
            metric = "exact_mae_y"
        else:
            metric = "loss"

    if metric == "loss":
        return (
            selected_by_loss,
            f"{loss_metric_label}+0.35*worst_block",
            float(pass_scores_by_loss[selected_by_loss]),
            {str(k): float(v) for k, v in pass_scores_by_loss.items()},
        )

    metric_extractors = {
        "exact_mae_y": ("exact.mean_abs_error_y", lambda s: float(s["mean_abs_error_y"])),
        "exact_rmse_y": ("exact.rmse_y", lambda s: float(s["rmse_y"])),
        "exact_abs_y0": ("exact.abs_error_mean_y0", lambda s: float(s["abs_error_mean_y0"])),
    }
    if metric not in metric_extractors:
        raise ValueError(
            f"Unsupported selection_metric='{selection_metric}'. "
            "Supported: auto, loss, exact_mae_y, exact_rmse_y, exact_abs_y0"
        )
    if len(exact_summary_by_pass) == 0:
        raise RuntimeError(
            f"selection_metric='{metric}' requires exact_solution metrics, but none are available"
        )

    label, extractor = metric_extractors[metric]
    scores = {}
    for pass_id, summary in exact_summary_by_pass.items():
        scores[int(pass_id)] = float(extractor(summary))
    selected_pass = int(min(scores, key=scores.get))
    return (
        selected_pass,
        label,
        float(scores[selected_pass]),
        {str(k): float(v) for k, v in scores.items()},
    )


def predict_recursive_stitched(
    block_blobs: List[Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    Xi_initial: np.ndarray,
    params: Dict[str, np.ndarray],
    N_per_block: int,
    D: int,
    layers: List[int],
    T_total: float,
    rollout_inputs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    if len(blocks) == 0:
        raise ValueError("blocks must contain at least one block")
    if Xi_initial.ndim != 2 or Xi_initial.shape[1] != D:
        raise ValueError(f"Xi_initial must have shape [M, {D}]")
    if rollout_inputs is not None and len(rollout_inputs) != len(blocks):
        raise ValueError("rollout_inputs must have one (t, W) pair per block")

    Xi_curr = Xi_initial.astype(np.float32)
    t_segments = []
    X_segments = []
    Y_segments = []
    Z_segments = []

    for b, block in enumerate(blocks):
        blob = block_blobs[b]
        tf.reset_default_graph()

        model = NN_Quadratic_Coupled_Recursive(
            Xi_generator=make_empirical_generator(Xi_curr, jitter_scale=0.0),
            T=block["T_block"],
            M=Xi_curr.shape[0],
            N=N_per_block,
            D=D,
            layers=layers,
            parameters=params,
            t_start=block["t_start"],
            t_end=block["t_end"],
            T_total=T_total,
            terminal_blob=None,
            normalize_time_input=bool(int(blob.get("normalize_time_input", 1))),
            x_norm_mean=blob.get("x_norm_mean", np.zeros((1, D), dtype=np.float32)),
            x_norm_std=blob.get("x_norm_std", np.ones((1, D), dtype=np.float32)),
        )

        try:
            model.import_parameter_blob(blob, strict=True)
            if rollout_inputs is None:
                t_b, W_b, _ = model.fetch_minibatch()
            else:
                t_b, W_b = rollout_inputs[b]
            X_b, Y_b, Z_b = model.predict(Xi_curr, t_b, W_b, const_value=1.0)

            start_idx = 0 if b == 0 else 1
            t_segments.append(t_b[:, start_idx:, :].astype(np.float32))
            X_segments.append(X_b[:, start_idx:, :].astype(np.float32))
            Y_segments.append(Y_b[:, start_idx:, :].astype(np.float32))
            Z_segments.append(Z_b[:, start_idx:, :].astype(np.float32))

            Xi_curr = X_b[:, -1, :].astype(np.float32)
        finally:
            model.sess.close()

    return {
        "t": np.concatenate(t_segments, axis=1),
        "X": np.concatenate(X_segments, axis=1),
        "Y": np.concatenate(Y_segments, axis=1),
        "Z": np.concatenate(Z_segments, axis=1),
    }


def _z_component_labels(n_components: int) -> List[str]:
    base = ["Z_S", "Z_H", "Z_V", "Z_X"]
    labels = []
    for i in range(int(n_components)):
        labels.append(base[i] if i < len(base) else f"Z_{i}")
    return labels


def build_exact_solution_functions(
    solution_name: str,
    params: Dict[str, np.ndarray],
    D: int,
) -> Optional[Dict[str, Any]]:
    name = str(solution_name or "none").strip().lower()
    if name in ("", "none", "off", "false", "0"):
        return None

    if name in ("quadratic_coupled", "quadratic", "qc4d"):
        if int(D) != 4:
            raise ValueError(
                f"exact_solution='quadratic_coupled' requires D=4, found D={int(D)}"
            )

        gamma = float(params["gamma"])
        s1 = float(params["s1"])
        s3 = float(params["s3"])

        def u_exact(t_arr: np.ndarray, Xi_arr: np.ndarray) -> np.ndarray:
            _ = t_arr
            Xi_arr = np.asarray(Xi_arr, dtype=np.float32)
            S = Xi_arr[:, 0:1]
            V = Xi_arr[:, 2:3]
            X_state = Xi_arr[:, 3:4]
            return (-gamma * np.exp(S) * X_state + V ** 2 + V * X_state).astype(np.float32)

        def z_exact(t_arr: np.ndarray, Xi_arr: np.ndarray) -> np.ndarray:
            _ = t_arr
            Xi_arr = np.asarray(Xi_arr, dtype=np.float32)
            S = Xi_arr[:, 0:1]
            V = Xi_arr[:, 2:3]
            X_state = Xi_arr[:, 3:4]

            z_s = -gamma * np.exp(S) * X_state * s1
            z_h = np.zeros_like(z_s)
            z_v = (2.0 * V + X_state) * s3
            z_x = np.zeros_like(z_s)
            return np.concatenate([z_s, z_h, z_v, z_x], axis=1).astype(np.float32)

        return {
            "name": "quadratic_coupled",
            "u_exact": u_exact,
            "z_exact": z_exact,
        }

    raise ValueError(
        "Unknown exact_solution profile "
        f"'{solution_name}'. Supported: none, quadratic_coupled"
    )


def compute_stitched_exact_bundle(
    stitched: Dict[str, np.ndarray],
    exact_solution: Dict[str, Any],
    eps: float = 1.0e-8,
) -> Dict[str, Any]:
    t_all = stitched["t"]
    X_all = stitched["X"]
    Y_pred = stitched["Y"]
    Z_pred = stitched["Z"]

    M_paths = int(X_all.shape[0])
    T_points = int(X_all.shape[1])
    D = int(X_all.shape[2])

    X_flat = X_all.reshape(-1, D)
    t_flat = t_all.reshape(-1, 1)

    Y_exact = exact_solution["u_exact"](t_flat, X_flat).reshape(M_paths, T_points, 1).astype(np.float32)
    Z_exact = exact_solution["z_exact"](t_flat, X_flat).reshape(M_paths, T_points, D).astype(np.float32)

    abs_err_Y = np.abs(Y_pred - Y_exact)
    abs_err_Z = np.abs(Z_pred - Z_exact)

    # Legacy relative error (kept for backward compatibility / diagnostics).
    rel_err_Z_legacy = abs_err_Z / (np.abs(Z_exact) + float(eps))

    # Robust relative error: ignore components where exact Z is (near) zero.
    valid_mask = np.abs(Z_exact) > float(eps)
    rel_err_Z = np.zeros_like(abs_err_Z, dtype=np.float32)
    np.divide(
        abs_err_Z,
        np.abs(Z_exact) + float(eps),
        out=rel_err_Z,
        where=valid_mask,
    )

    y0_pred = Y_pred[:, 0, 0]
    y0_exact = Y_exact[:, 0, 0]

    mean_abs_err_Y_t = np.mean(abs_err_Y[:, :, 0], axis=0)
    mean_abs_err_Z_t = np.mean(abs_err_Z, axis=0)
    mean_rel_err_Z_legacy_t = np.mean(rel_err_Z_legacy, axis=0)
    valid_count_t = np.maximum(np.sum(valid_mask, axis=0), 1.0).astype(np.float32)
    mean_rel_err_Z_t = (np.sum(rel_err_Z, axis=0) / valid_count_t).astype(np.float32)
    valid_count_all = float(max(np.sum(valid_mask), 1.0))
    valid_count_comp = np.maximum(np.sum(valid_mask, axis=(0, 1)), 1.0).astype(np.float32)

    summary = {
        "solution_name": exact_solution.get("name", "unknown"),
        "n_paths": int(M_paths),
        "n_time_points": int(T_points),
        "mean_pred_y0": float(np.mean(y0_pred)),
        "mean_exact_y0": float(np.mean(y0_exact)),
        "abs_error_mean_y0": float(np.mean(np.abs(y0_pred - y0_exact))),
        "rmse_y0": float(np.sqrt(np.mean((y0_pred - y0_exact) ** 2))),
        "mean_abs_error_y": float(np.mean(abs_err_Y)),
        "rmse_y": float(np.sqrt(np.mean((Y_pred - Y_exact) ** 2))),
        "mean_abs_error_z": float(np.mean(abs_err_Z)),
        "mean_rel_error_z": float(np.sum(rel_err_Z) / valid_count_all),
        "mean_rel_error_z_legacy": float(np.mean(rel_err_Z_legacy)),
        "mean_abs_error_z_by_component": np.mean(abs_err_Z, axis=(0, 1)).astype(np.float32),
        "mean_rel_error_z_by_component": (np.sum(rel_err_Z, axis=(0, 1)) / valid_count_comp).astype(np.float32),
        "mean_rel_error_z_by_component_legacy": np.mean(rel_err_Z_legacy, axis=(0, 1)).astype(np.float32),
        "valid_rel_error_fraction_z_by_component": np.mean(valid_mask.astype(np.float32), axis=(0, 1)).astype(np.float32),
        "z_component_labels": _z_component_labels(D),
    }

    timeseries = {
        "t": t_all[0, :, 0].astype(np.float32),
        "mean_abs_error_y": mean_abs_err_Y_t.astype(np.float32),
        "mean_abs_error_z": mean_abs_err_Z_t.astype(np.float32),
        "mean_rel_error_z": mean_rel_err_Z_t.astype(np.float32),
        "mean_rel_error_z_legacy": mean_rel_err_Z_legacy_t.astype(np.float32),
        "valid_rel_error_fraction_z": np.mean(valid_mask.astype(np.float32), axis=0).astype(np.float32),
        "z_component_labels": _z_component_labels(D),
    }

    return {
        "summary": summary,
        "timeseries": timeseries,
        "Y_exact": Y_exact,
        "Z_exact": Z_exact,
    }


def save_exact_error_timeseries_csv(timeseries: Dict[str, np.ndarray], path: str) -> None:
    t = np.asarray(timeseries["t"])
    abs_y = np.asarray(timeseries["mean_abs_error_y"])
    abs_z = np.asarray(timeseries["mean_abs_error_z"])
    rel_z = np.asarray(timeseries["mean_rel_error_z"])
    rel_z_legacy = np.asarray(timeseries["mean_rel_error_z_legacy"]) if "mean_rel_error_z_legacy" in timeseries else None
    labels = timeseries.get("z_component_labels", _z_component_labels(abs_z.shape[1]))

    rows = []
    for i in range(int(t.shape[0])):
        row = {
            "t": float(t[i]),
            "mean_abs_error_y": float(abs_y[i]),
        }
        for d, label in enumerate(labels):
            row[f"mean_abs_error_{label}"] = float(abs_z[i, d])
            row[f"mean_rel_error_{label}"] = float(rel_z[i, d])
            if rel_z_legacy is not None:
                row[f"mean_rel_error_{label}_legacy"] = float(rel_z_legacy[i, d])
        rows.append(row)

    save_rows_csv(rows, path)


def plot_recursive_exact_comparison(
    stitched: Dict[str, np.ndarray],
    Y_exact: np.ndarray,
    Z_exact: np.ndarray,
    blocks: List[Dict[str, float]],
    out_dir: str,
    sample_paths: int = 5,
    file_suffix: str = "",
) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_exact_comparison")
        return
    if stitched is None:
        return
    if "t" not in stitched or "Y" not in stitched or "Z" not in stitched:
        return

    t_all = stitched["t"]
    Y_pred = stitched["Y"]
    Z_pred = stitched["Z"]
    if t_all.size == 0 or Y_pred.size == 0 or Z_pred.size == 0:
        return
    if Y_exact.shape != Y_pred.shape or Z_exact.shape != Z_pred.shape:
        return

    os.makedirs(out_dir, exist_ok=True)
    n_paths = max(1, min(int(sample_paths), int(t_all.shape[0])))
    z_labels = _z_component_labels(int(Z_pred.shape[2]))
    z_colors = ["b", "g", "r", "m", "c", "y", "k"]

    plt.figure(figsize=(12, 6))
    for i in range(n_paths):
        alpha = 0.95 if i == 0 else 0.28
        width = 1.8 if i == 0 else 0.9
        pred_label = "Y pred" if i == 0 else None
        exact_label = "Y exact" if i == 0 else None
        plt.plot(
            t_all[i, :, 0],
            Y_pred[i, :, 0],
            color="tab:blue",
            alpha=alpha,
            linewidth=width,
            label=pred_label,
        )
        plt.plot(
            t_all[i, :, 0],
            Y_exact[i, :, 0],
            color="tab:red",
            alpha=alpha,
            linewidth=width,
            linestyle="--",
            label=exact_label,
        )
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)
    plt.title("Recursive stitched prediction - Y predicted vs exact")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"recursive_stitched_Y_exact{file_suffix}.png"), dpi=160)
    plt.close()

    abs_err_Z_full = np.abs(Z_pred - Z_exact)
    valid_mask = np.abs(Z_exact) > 1.0e-8
    rel_err_Z = np.zeros_like(abs_err_Z_full, dtype=np.float32)
    np.divide(
        abs_err_Z_full,
        np.abs(Z_exact) + 1.0e-8,
        out=rel_err_Z,
        where=valid_mask,
    )
    valid_count_t = np.maximum(np.sum(valid_mask, axis=0), 1.0).astype(np.float32)
    mean_rel_err_Z = np.sum(rel_err_Z, axis=0) / valid_count_t

    plt.figure(figsize=(12, 6))
    for d in range(Z_pred.shape[2]):
        color = z_colors[d % len(z_colors)]
        label = z_labels[d] if d < len(z_labels) else f"Z[{d}]"
        curve = np.maximum(mean_rel_err_Z[:, d], 1.0e-14)
        plt.plot(t_all[0, :, 0], curve, color=color, linewidth=1.5, label=f"Mean rel err {label}")
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)
    plt.yscale("log")
    plt.title("Recursive stitched prediction - Mean relative error on Z")
    plt.xlabel("Time")
    plt.ylabel("Relative error (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"recursive_stitched_Z_rel_error{file_suffix}.png"), dpi=160)
    plt.close()

    abs_err_Z = np.mean(abs_err_Z_full, axis=0)
    abs_err_Y = np.mean(np.abs(Y_pred - Y_exact), axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(
        t_all[0, :, 0],
        np.maximum(abs_err_Y[:, 0], 1.0e-14),
        color="tab:orange",
        linewidth=1.8,
        label="Mean abs err Y",
    )
    for d in range(Z_pred.shape[2]):
        color = z_colors[d % len(z_colors)]
        label = z_labels[d] if d < len(z_labels) else f"Z[{d}]"
        plt.plot(
            t_all[0, :, 0],
            np.maximum(abs_err_Z[:, d], 1.0e-14),
            color=color,
            linewidth=1.4,
            label=f"Mean abs err {label}",
        )
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)
    plt.yscale("log")
    plt.title("Recursive stitched prediction - Mean absolute error on Y and Z")
    plt.xlabel("Time")
    plt.ylabel("Absolute error (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"recursive_stitched_abs_error{file_suffix}.png"), dpi=160)
    plt.close()


def plot_recursive_stitched_predictions(
    stitched: Dict[str, np.ndarray],
    blocks: List[Dict[str, float]],
    out_dir: str,
    sample_paths: int = 5,
    file_suffix: str = "",
) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_stitched_predictions")
        return
    if stitched is None:
        return
    if "t" not in stitched or "X" not in stitched or "Y" not in stitched:
        return

    t_all = stitched["t"]
    X_all = stitched["X"]
    Y_all = stitched["Y"]
    if t_all.size == 0 or X_all.size == 0 or Y_all.size == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    n_paths = max(1, min(int(sample_paths), int(t_all.shape[0])))

    component_labels = ["S", "H", "V", "X"]
    component_colors = ["b", "r", "y", "g"]

    plt.figure(figsize=(12, 6))
    for d in range(X_all.shape[2]):
        label = component_labels[d] if d < len(component_labels) else f"X[{d}]"
        color = component_colors[d % len(component_colors)]
        plt.plot(t_all[0, :, 0], X_all[0, :, d], color=color, linewidth=1.5, label=label)
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.3)
    plt.title("Recursive stitched prediction - State path (single continuous horizon)")
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"recursive_stitched_state_path{file_suffix}.png"),
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    for i in range(n_paths):
        alpha = 0.95 if i == 0 else 0.35
        width = 1.8 if i == 0 else 1.0
        label = "Y pred (path 0)" if i == 0 else None
        plt.plot(t_all[i, :, 0], Y_all[i, :, 0], color="tab:blue", alpha=alpha, linewidth=width, label=label)
    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.3)
    plt.title("Recursive stitched prediction - Y over full horizon")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    if n_paths > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"recursive_stitched_Y_pred{file_suffix}.png"),
        dpi=160,
    )
    plt.close()


def plot_recursive_stitched_y_convergence(
    stitched_by_pass: Dict[int, Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    out_dir: str,
    sample_paths: int = 8,
) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_stitched_y_convergence")
        return
    if stitched_by_pass is None or len(stitched_by_pass) == 0:
        return

    pass_ids = sorted(stitched_by_pass.keys())
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(pass_ids)))

    plt.figure(figsize=(12, 6))
    for i, pass_id in enumerate(pass_ids):
        stitched = stitched_by_pass[pass_id]
        t_all = stitched.get("t", None)
        Y_all = stitched.get("Y", None)
        if t_all is None or Y_all is None or t_all.size == 0 or Y_all.size == 0:
            continue
        n_paths = max(1, min(int(sample_paths), int(t_all.shape[0])))
        t_flat = t_all[:n_paths, :, 0].reshape(-1)
        y_flat = Y_all[:n_paths, :, 0].reshape(-1)
        plt.scatter(t_flat, y_flat, s=2, color=colors[i], alpha=0.06)
        y_mean = np.mean(Y_all[:n_paths, :, 0], axis=0)
        plt.plot(
            t_all[0, :, 0],
            y_mean,
            color=colors[i],
            linewidth=1.8,
            label=f"{_pass_label(pass_id)} mean Y",
        )

    for block in blocks[:-1]:
        plt.axvline(float(block["t_end"]), color="k", linestyle="--", linewidth=0.8, alpha=0.25)

    plt.title("Recursive stitched prediction - Y convergence across passes")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_stitched_Y_convergence.png"), dpi=160)
    plt.close()


class NN_Quadratic_Coupled_Recursive(NN_Quadratic_Coupled):
    """
    Wrapper per training a blocchi:
    - tempi assoluti nel blocco [t_start, t_end]
    - input time normalization opzionale
    - g terminale sostituita con u del blocco successivo congelato (via blob)
    """

    def __init__(
        self,
        Xi_generator,
        T,
        M,
        N,
        D,
        layers,
        parameters,
        t_start,
        t_end,
        T_total,
        terminal_blob=None,
        normalize_time_input=True,
        x_norm_mean=None,
        x_norm_std=None,
    ):
        self.t_start = np.float32(t_start)
        self.t_end = np.float32(t_end)
        self.T_total = np.float32(T_total)
        self.normalize_time_input = bool(normalize_time_input)

        x_mean = np.zeros((1, D), dtype=np.float32) if x_norm_mean is None else np.asarray(x_norm_mean, dtype=np.float32).reshape(1, D)
        x_std = np.ones((1, D), dtype=np.float32) if x_norm_std is None else np.asarray(x_norm_std, dtype=np.float32).reshape(1, D)
        self.x_norm_mean_np = x_mean
        self.x_norm_std_np = np.maximum(x_std, 1.0e-3).astype(np.float32)

        self.terminal_blob = _as_blob_dict(terminal_blob)
        self._terminal_weights_tf = None
        self._terminal_biases_tf = None
        self._terminal_x_mean_tf = None
        self._terminal_x_std_tf = None
        self._terminal_T_total_tf = None
        self._terminal_use_time = False

        # Costanti TF create PRIMA di super().__init__ perché net_u viene usata nella build del grafo.
        self._x_norm_mean_tf = tf.constant(self.x_norm_mean_np, dtype=tf.float32)
        self._x_norm_std_tf = tf.constant(self.x_norm_std_np, dtype=tf.float32)
        self._T_total_tf = tf.constant(self.T_total, dtype=tf.float32)

        super().__init__(Xi_generator, T, M, N, D, layers, parameters)

    def _normalize_t(self, t):
        if not self.normalize_time_input:
            return t
        return 2.0 * (t / self._T_total_tf) - 1.0

    def _normalize_x(self, X):
        return (X - self._x_norm_mean_tf) / self._x_norm_std_tf

    def net_u(self, t, X):
        t_in = self._normalize_t(t)
        X_in = self._normalize_x(X)
        u = self.neural_net(tf.concat([t_in, X_in], 1), self.weights, self.biases)
        Du = tf.gradients(u, X)[0]
        return u, Du

    def fetch_minibatch(self):
        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1), dtype=np.float32)
        DW = np.zeros((M, N + 1, D), dtype=np.float32)
        dt = self.T / N

        Dt[:, 1:, :] = dt
        if self.use_antithetic_sampling and M > 1:
            half_M = M // 2
            DW_half = np.sqrt(dt) * np.random.normal(size=(half_M, N, D))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M : 2 * half_M, 1:, :] = -DW_half
            if M % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * np.random.normal(size=(N, D))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = self.t_start + np.cumsum(Dt, axis=1)  # tempi assoluti nel blocco
        W = np.cumsum(DW, axis=1)
        Xi_batch = self.Xi_generator(M, D).astype(np.float32)
        return t, W, Xi_batch

    def _build_terminal_constants_if_needed(self):
        if self.terminal_blob is None:
            return
        if self._terminal_weights_tf is not None:
            return

        n_layers = int(self.terminal_blob["n_layers"])
        self._terminal_weights_tf = []
        self._terminal_biases_tf = []
        for i in range(n_layers):
            self._terminal_weights_tf.append(
                tf.constant(self.terminal_blob[f"W_{i}"], dtype=tf.float32)
            )
            self._terminal_biases_tf.append(
                tf.constant(self.terminal_blob[f"b_{i}"], dtype=tf.float32)
            )

        self._terminal_x_mean_tf = tf.constant(
            self.terminal_blob.get("x_norm_mean", np.zeros((1, self.D), dtype=np.float32)),
            dtype=tf.float32,
        )
        self._terminal_x_std_tf = tf.constant(
            np.maximum(
                self.terminal_blob.get("x_norm_std", np.ones((1, self.D), dtype=np.float32)),
                1.0e-3,
            ),
            dtype=tf.float32,
        )
        self._terminal_T_total_tf = tf.constant(
            np.float32(self.terminal_blob.get("T_total", self.T_total)), dtype=tf.float32
        )
        self._terminal_use_time = bool(int(self.terminal_blob.get("normalize_time_input", 1)))

    def _terminal_u(self, t_abs, X):
        self._build_terminal_constants_if_needed()
        t_in = t_abs
        if self._terminal_use_time:
            t_in = 2.0 * (t_abs / self._terminal_T_total_tf) - 1.0
        X_in = (X - self._terminal_x_mean_tf) / self._terminal_x_std_tf
        return self.neural_net(
            tf.concat([t_in, X_in], 1), self._terminal_weights_tf, self._terminal_biases_tf
        )

    def g_tf(self, X):
        if self.terminal_blob is None:
            return super().g_tf(X)
        t_eval = tf.ones([tf.shape(X)[0], 1], dtype=tf.float32) * tf.constant(
            self.t_end, dtype=tf.float32
        )
        return self._terminal_u(t_eval, X)

    def Dg_tf(self, X):
        if self.terminal_blob is None:
            return super().Dg_tf(X)
        return tf.gradients(self.g_tf(X), X)[0]

    def export_parameter_blob(self) -> Dict[str, np.ndarray]:
        values = self.sess.run(self.weights + self.biases)
        n_layers = len(self.weights)
        blob = {
            "n_layers": np.array(n_layers, dtype=np.int32),
            "layers": np.asarray(self.layers, dtype=np.int32),
            "t_start": np.asarray(self.t_start, dtype=np.float32),
            "t_end": np.asarray(self.t_end, dtype=np.float32),
            "T_total": np.asarray(self.T_total, dtype=np.float32),
            "normalize_time_input": np.asarray(int(self.normalize_time_input), dtype=np.int32),
            "x_norm_mean": np.asarray(self.x_norm_mean_np, dtype=np.float32),
            "x_norm_std": np.asarray(self.x_norm_std_np, dtype=np.float32),
        }
        for i in range(n_layers):
            blob[f"W_{i}"] = values[i].astype(np.float32)
            blob[f"b_{i}"] = values[n_layers + i].astype(np.float32)
        return blob

    def import_parameter_blob(self, blob_or_path, strict=True):
        blob = _as_blob_dict(blob_or_path)
        if blob is None:
            return
        n_layers = len(self.weights)
        if strict and int(blob["n_layers"]) != n_layers:
            raise ValueError(
                f"n_layers mismatch: model={n_layers}, blob={int(blob['n_layers'])}"
            )
        for i in range(n_layers):
            w_key = f"W_{i}"
            b_key = f"b_{i}"
            if w_key in blob:
                self.sess.run(self.weights[i].assign(blob[w_key]))
            elif strict:
                raise KeyError(f"Missing key {w_key} in blob")
            if b_key in blob:
                self.sess.run(self.biases[i].assign(blob[b_key]))
            elif strict:
                raise KeyError(f"Missing key {b_key} in blob")

    def save_parameter_blob(self, path: str) -> None:
        save_blob_npz(self.export_parameter_blob(), path)

    def load_parameter_blob(self, path: str, strict=True) -> None:
        self.import_parameter_blob(path, strict=strict)


###############################################################################
# Utilità training standard + ricorsivo
###############################################################################


def Xi_generator_default(M, D):
    assert D == 4
    Xi = np.zeros((M, 4), dtype=np.float32)
    Xi[:, 0] = np.random.normal(1.0, 1.0, M)
    Xi[:, 1] = np.random.normal(1.0, 1.0, M)
    Xi[:, 2] = np.random.normal(0.0, 1.0, M)
    Xi[:, 3] = np.random.uniform(1.0, 9.0, M)
    return Xi.astype(np.float32)


def make_empirical_generator(samples: np.ndarray, jitter_scale: float = 0.0):
    samples = np.asarray(samples, dtype=np.float32)
    mean = np.mean(samples, axis=0, keepdims=True)
    std = np.std(samples, axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-3)

    def _gen(M, D):
        idx = np.random.randint(0, samples.shape[0], size=M)
        Xi = samples[idx].copy()
        if jitter_scale > 0.0:
            Xi += jitter_scale * std * np.random.normal(size=Xi.shape).astype(np.float32)
        return Xi.astype(np.float32)

    return _gen


def estimate_generator_stats(generator_fn, D, n_samples=4096):
    x = generator_fn(n_samples, D).astype(np.float32)
    mean = np.mean(x, axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(np.std(x, axis=0, keepdims=True), 1.0e-3).astype(np.float32)
    return mean, std


def build_blocks(T_total: float, block_size: float) -> List[Dict[str, float]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    n_blocks = int(np.ceil(T_total / block_size))
    edges = [0.0]
    for i in range(1, n_blocks + 1):
        edges.append(min(float(i * block_size), float(T_total)))
    blocks = []
    for i in range(n_blocks):
        t0 = float(edges[i])
        t1 = float(edges[i + 1])
        blocks.append({"idx": i, "t_start": t0, "t_end": t1, "T_block": (t1 - t0)})
    return blocks


def _normalize_training_plan_rule(raw: Dict[str, Any], source_row: int = 0) -> Optional[Dict]:
    if raw is None:
        return None

    pass_scope = str(raw.get("pass_scope", "")).strip()
    block_scope = str(raw.get("block_scope", "")).strip().lower()
    phase = str(raw.get("phase", "")).strip().lower()
    if pass_scope == "" or block_scope == "" or phase == "":
        return None
    if phase not in ("stage", "final", "refine"):
        raise ValueError(
            f"Invalid phase '{phase}' in training plan rule (allowed: stage, final, refine)"
        )

    enabled_raw = str(raw.get("enabled", "1")).strip().lower()
    enabled = enabled_raw not in ("0", "false", "no", "off", "")
    if not enabled:
        return None

    order_raw = str(raw.get("order", "0")).strip()
    order = int(order_raw) if order_raw != "" else 0
    n_iter = int(str(raw.get("n_iter", "")).strip())
    lr = float(str(raw.get("lr", "")).strip())
    if n_iter <= 0:
        raise ValueError(f"n_iter must be > 0 in training plan rule (source_row={source_row})")
    if lr <= 0:
        raise ValueError(f"lr must be > 0 in training plan rule (source_row={source_row})")

    return {
        "pass_scope": pass_scope,
        "block_scope": block_scope,
        "phase": phase,
        "order": int(order),
        "n_iter": int(n_iter),
        "lr": float(lr),
        "source_row": int(source_row),
    }


def load_training_plan_csv(csv_path: Optional[str]) -> List[Dict]:
    if csv_path is None or str(csv_path).strip() == "":
        return []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training plan CSV not found: {csv_path}")

    rules = []
    required = {"pass_scope", "block_scope", "phase", "n_iter", "lr"}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Training plan CSV is empty or has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Training plan CSV missing required columns: {sorted(missing)}")

        for i_row, row in enumerate(reader, start=2):
            normalized = _normalize_training_plan_rule(row, source_row=i_row)
            if normalized is not None:
                rules.append(normalized)

    return rules


def _find_resume_run_root(resume_models_dir: str) -> Optional[str]:
    resume_models_dir = str(resume_models_dir or "").strip()
    if resume_models_dir == "":
        return None

    p = os.path.abspath(os.path.expanduser(resume_models_dir))
    candidates = []
    cur = p
    for _ in range(5):
        candidates.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        cfg = os.path.join(c, "run_config.json")
        if os.path.isfile(cfg):
            return c
    return None


def load_training_plan_rules_from_resume_run(
    resume_models_dir: str,
) -> Tuple[List[Dict], Optional[str], Optional[str]]:
    run_root = _find_resume_run_root(resume_models_dir)
    if run_root is None:
        return [], None, None

    cfg_path = os.path.join(run_root, "run_config.json")
    if not os.path.isfile(cfg_path):
        return [], None, None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_rules = cfg.get("training_plan_rules", [])
    plan_csv = cfg.get("training_plan_csv", None)
    if not isinstance(raw_rules, list):
        return [], cfg_path, plan_csv

    rules = []
    for i, raw in enumerate(raw_rules):
        normalized = _normalize_training_plan_rule(raw, source_row=int(raw.get("source_row", i + 1)))
        if normalized is not None:
            rules.append(normalized)
    return rules, cfg_path, plan_csv


def _find_resume_eval_bundle_path(resume_models_dir: str) -> Optional[str]:
    run_root = _find_resume_run_root(resume_models_dir)
    if run_root is None:
        return None
    candidate = os.path.join(run_root, "recursive", "evaluation_bundle.npz")
    if os.path.isfile(candidate):
        return candidate
    return None


def _pass_scope_priority(pass_scope: str, pass_id: int) -> int:
    ps = str(pass_scope).strip().lower()
    if ps in ("*", "all"):
        return 1

    if ps.endswith("+"):
        base = ps[:-1].strip()
        if base.isdigit() and pass_id >= int(base):
            return 2

    if ps.startswith(">="):
        base = ps[2:].strip()
        if base.isdigit() and pass_id >= int(base):
            return 2

    if ps.isdigit() and pass_id == int(ps):
        return 3

    return -1


def _block_scope_priority(block_scope: str, block_idx: int, n_blocks: int) -> int:
    bs = str(block_scope).strip().lower()
    is_terminal = block_idx == (n_blocks - 1)

    if bs in ("*", "all"):
        return 1
    if bs == "terminal" and is_terminal:
        return 2
    if bs == "other" and (not is_terminal):
        return 2

    if bs.startswith("block:"):
        token = bs.split(":", 1)[1].strip()
        if token.isdigit() and block_idx == int(token):
            return 3
    if bs.startswith("idx:"):
        token = bs.split(":", 1)[1].strip()
        if token.isdigit() and block_idx == int(token):
            return 3

    if bs.isdigit() and block_idx == int(bs):
        return 3

    return -1


def _resolve_phase_plan(
    rules: List[Dict],
    phase: str,
    pass_id: int,
    block_idx: int,
    n_blocks: int,
    default_plan: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    matched = []
    for r in rules:
        if r["phase"] != phase:
            continue
        p_prio = _pass_scope_priority(r["pass_scope"], pass_id)
        if p_prio < 0:
            continue
        b_prio = _block_scope_priority(r["block_scope"], block_idx, n_blocks)
        if b_prio < 0:
            continue
        matched.append((p_prio, b_prio, r["order"], r))

    if len(matched) == 0:
        return list(default_plan)

    best_scope = max((x[0], x[1]) for x in matched)
    selected = [x for x in matched if (x[0], x[1]) == best_scope]
    selected.sort(key=lambda x: x[2])
    return [(int(x[3]["n_iter"]), float(x[3]["lr"])) for x in selected]


def resolve_training_plan_for_block(
    rules: List[Dict],
    pass_id: int,
    block_idx: int,
    n_blocks: int,
    default_stage: List[Tuple[int, float]],
    default_final: List[Tuple[int, float]],
    default_refine: List[Tuple[int, float]],
) -> Dict[str, List[Tuple[int, float]]]:
    if rules is None:
        rules = []
    return {
        "stage_plan": _resolve_phase_plan(
            rules=rules,
            phase="stage",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_stage,
        ),
        "final_plan": _resolve_phase_plan(
            rules=rules,
            phase="final",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_final,
        ),
        "refine_plan": _resolve_phase_plan(
            rules=rules,
            phase="refine",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_refine,
        ),
    }


def train_with_standard_schedule(
    model: FBSNN,
    stage_plan: List[Tuple[int, float]],
    final_plan: List[Tuple[int, float]],
    eval_batches=5,
    precision_target: Optional[float] = None,
    max_refine_rounds: int = 3,
    refine_plan: Optional[List[Tuple[int, float]]] = None,
    label: str = "",
):
    stage_logs = []

    coupling_step = 1.0
    coupling_levels = np.arange(1.0, 1.0 + coupling_step, coupling_step, dtype=np.float32)

    for level in coupling_levels:
        model.const = np.float32(level)
        print(f"=== [{label}] Coupling stage: const={float(level):.1f} ===")
        for n_iter, lr in stage_plan:
            t0 = time.time()
            train_stats = model.train(N_Iter=n_iter, learning_rate=lr, const_value=level)
            eval_stats = model.evaluate(const_value=level, n_batches=eval_batches)
            elapsed = time.time() - t0
            stage_logs.append(
                {
                    "phase": "curriculum",
                    "const": float(level),
                    "lr": float(lr),
                    "n_iter": int(n_iter),
                    "train_last_loss": train_stats["last_loss"],
                    "eval_mean_loss": eval_stats["mean_loss"],
                    "eval_std_loss": eval_stats["std_loss"],
                    "eval_mean_loss_per_sample": eval_stats["mean_loss_per_sample"],
                    "eval_std_loss_per_sample": eval_stats["std_loss_per_sample"],
                    "eval_mean_y0": eval_stats["mean_y0"],
                    "eval_std_y0": eval_stats["std_y0"],
                    "elapsed_sec": float(elapsed),
                }
            )
            print(
                f"[StageSummary] {label} const={level:.1f}, lr={lr:.1e}, iters={n_iter}, "
                f"eval_loss={eval_stats['mean_loss']:.3e}±{eval_stats['std_loss']:.2e}, "
                f"eval_Y0={eval_stats['mean_y0']:.3f}±{eval_stats['std_y0']:.3f}, time={elapsed:.1f}s"
            )

    model.const = np.float32(1.0)
    print(f"=== [{label}] Final fine-tuning at const=1.0 ===")
    for n_iter, lr in final_plan:
        t0 = time.time()
        train_stats = model.train(
            N_Iter=n_iter,
            learning_rate=lr,
            const_value=1.0,
            eval_every=25,
            val_batches=8,
            early_stopping_metric="loss",
            patience=150,
            min_delta=1e-3,
            restore_best=True,
        )
        eval_stats = model.evaluate(const_value=1.0, n_batches=eval_batches)
        elapsed = time.time() - t0
        stage_logs.append(
            {
                "phase": "final_finetune",
                "const": 1.0,
                "lr": float(lr),
                "n_iter": int(n_iter),
                "train_last_loss": train_stats["last_loss"],
                "best_iter": train_stats["best_iter"],
                "best_score": train_stats["best_score"],
                "stopped_early": train_stats["stopped_early"],
                "eval_mean_loss": eval_stats["mean_loss"],
                "eval_std_loss": eval_stats["std_loss"],
                "eval_mean_loss_per_sample": eval_stats["mean_loss_per_sample"],
                "eval_std_loss_per_sample": eval_stats["std_loss_per_sample"],
                "eval_mean_y0": eval_stats["mean_y0"],
                "eval_std_y0": eval_stats["std_y0"],
                "elapsed_sec": float(elapsed),
            }
        )
        print(
            f"[FinalSummary] {label} const=1.0, lr={lr:.1e}, iters={n_iter}, "
            f"best_it={train_stats['best_iter']}, best_score={train_stats['best_score']:.3e}, "
            f"eval_loss={eval_stats['mean_loss']:.3e}±{eval_stats['std_loss']:.2e}, "
            f"eval_Y0={eval_stats['mean_y0']:.3f}±{eval_stats['std_y0']:.3f}, time={elapsed:.1f}s"
        )

    eval_stats = model.evaluate(const_value=1.0, n_batches=eval_batches)
    refine_rounds = 0
    local_refine_plan = refine_plan if refine_plan is not None else [(50, 1e-5), (50, 5e-6)]

    while (
        precision_target is not None
        and eval_stats["mean_loss"] > precision_target
        and refine_rounds < max_refine_rounds
    ):
        refine_rounds += 1
        print(
            f"[Refine] {label} round={refine_rounds}, "
            f"loss={eval_stats['mean_loss']:.3e} > target={precision_target:.3e}"
        )
        for n_iter, lr in local_refine_plan:
            model.train(
                N_Iter=n_iter,
                learning_rate=lr,
                const_value=1.0,
                eval_every=25,
                val_batches=8,
                early_stopping_metric="loss",
                patience=100,
                min_delta=1e-3,
                restore_best=True,
            )
        eval_stats = model.evaluate(const_value=1.0, n_batches=eval_batches)

    return {
        "stage_logs": stage_logs,
        "eval_stats": eval_stats,
        "refine_rounds": int(refine_rounds),
        "precision_target": None if precision_target is None else float(precision_target),
    }


def run_standard_reference(
    Xi_generator,
    params,
    M,
    N,
    D,
    T,
    layers,
    stage_plan,
    final_plan,
):
    tf.reset_default_graph()
    model = NN_Quadratic_Coupled(Xi_generator, T, M, N, D, layers, params)
    logs = train_with_standard_schedule(
        model=model,
        stage_plan=stage_plan,
        final_plan=final_plan,
        eval_batches=5,
        precision_target=None,
        label="standard",
    )
    return model, logs


def _validate_loaded_blob_for_block(
    blob: Dict[str, np.ndarray],
    block: Dict[str, float],
    layers: List[int],
    D: int,
    T_total: float,
    pass_id: int,
    block_idx: int,
) -> None:
    if blob is None:
        raise ValueError(f"Empty blob for pass={pass_id}, block={block_idx}")

    model_n_layers = len(layers) - 1
    if "n_layers" in blob and int(blob["n_layers"]) != int(model_n_layers):
        raise ValueError(
            f"n_layers mismatch for pass={pass_id}, block={block_idx}: "
            f"blob={int(blob['n_layers'])}, expected={model_n_layers}"
        )

    if "layers" in blob:
        expected_layers = np.asarray(layers, dtype=np.int32)
        blob_layers = np.asarray(blob["layers"], dtype=np.int32)
        if expected_layers.shape != blob_layers.shape or not np.all(expected_layers == blob_layers):
            raise ValueError(
                f"layers mismatch for pass={pass_id}, block={block_idx}: "
                f"blob={blob_layers.tolist()}, expected={expected_layers.tolist()}"
            )

    if "T_total" in blob:
        blob_t_total = float(np.asarray(blob["T_total"]).reshape(-1)[0])
        if abs(blob_t_total - float(T_total)) > 1.0e-5:
            raise ValueError(
                f"T_total mismatch for pass={pass_id}, block={block_idx}: "
                f"blob={blob_t_total}, expected={float(T_total)}"
            )

    if "t_start" in blob:
        blob_t_start = float(np.asarray(blob["t_start"]).reshape(-1)[0])
        if abs(blob_t_start - float(block["t_start"])) > 1.0e-5:
            raise ValueError(
                f"t_start mismatch for pass={pass_id}, block={block_idx}: "
                f"blob={blob_t_start}, expected={float(block['t_start'])}"
            )
    if "t_end" in blob:
        blob_t_end = float(np.asarray(blob["t_end"]).reshape(-1)[0])
        if abs(blob_t_end - float(block["t_end"])) > 1.0e-5:
            raise ValueError(
                f"t_end mismatch for pass={pass_id}, block={block_idx}: "
                f"blob={blob_t_end}, expected={float(block['t_end'])}"
            )

    if "x_norm_mean" in blob:
        x_mean = np.asarray(blob["x_norm_mean"])
        if x_mean.ndim != 2 or x_mean.shape[1] != int(D):
            raise ValueError(
                f"x_norm_mean shape mismatch for pass={pass_id}, block={block_idx}: "
                f"blob={list(x_mean.shape)}, expected=[1,{int(D)}]"
            )
    if "x_norm_std" in blob:
        x_std = np.asarray(blob["x_norm_std"])
        if x_std.ndim != 2 or x_std.shape[1] != int(D):
            raise ValueError(
                f"x_norm_std shape mismatch for pass={pass_id}, block={block_idx}: "
                f"blob={list(x_std.shape)}, expected=[1,{int(D)}]"
            )


def detect_available_recursive_passes(models_dir: str) -> List[int]:
    if not os.path.isdir(models_dir):
        return []
    pass_ids = []
    for name in os.listdir(models_dir):
        full = os.path.join(models_dir, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith("pass_"):
            continue
        token = name.split("pass_", 1)[1]
        if token.isdigit():
            pass_ids.append(int(token))
    return sorted(pass_ids)


def resolve_resume_models_dir(resume_path: str) -> str:
    candidate_paths = [
        resume_path,
        os.path.join(resume_path, "models"),
        os.path.join(resume_path, "recursive", "models"),
    ]
    for c in candidate_paths:
        if len(detect_available_recursive_passes(c)) > 0:
            return c
    return resume_path


def load_pass_blobs_from_models_dir(
    models_dir: str,
    pass_id: int,
    blocks: List[Dict[str, float]],
    layers: List[int],
    D: int,
    T_total: float,
) -> List[Dict[str, np.ndarray]]:
    pass_dir = os.path.join(models_dir, f"pass_{pass_id}")
    if not os.path.isdir(pass_dir):
        raise FileNotFoundError(f"Missing directory for pass={pass_id}: {pass_dir}")

    loaded = []
    for b, block in enumerate(blocks):
        blob_path = os.path.join(pass_dir, f"block_{b:02d}.npz")
        if not os.path.exists(blob_path):
            raise FileNotFoundError(
                f"Missing blob for pass={pass_id}, block={b}: {blob_path}"
            )
        blob = _as_blob_dict(blob_path)
        _validate_loaded_blob_for_block(
            blob=blob,
            block=block,
            layers=layers,
            D=D,
            T_total=T_total,
            pass_id=pass_id,
            block_idx=b,
        )
        loaded.append(blob)
    return loaded


def rollout_boundaries(
    block_blobs: List[Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    Xi_generator,
    params,
    M_rollout,
    N_per_block,
    D,
    layers,
    T_total,
):
    boundary_samples = []
    Xi_curr = Xi_generator(M_rollout, D).astype(np.float32)
    boundary_samples.append(Xi_curr.copy())

    for b, block in enumerate(blocks):
        tf.reset_default_graph()
        model = NN_Quadratic_Coupled_Recursive(
            Xi_generator=make_empirical_generator(Xi_curr, jitter_scale=0.0),
            T=block["T_block"],
            M=M_rollout,
            N=N_per_block,
            D=D,
            layers=layers,
            parameters=params,
            t_start=block["t_start"],
            t_end=block["t_end"],
            T_total=T_total,
            terminal_blob=None,
            normalize_time_input=bool(int(block_blobs[b].get("normalize_time_input", 1))),
            x_norm_mean=block_blobs[b].get("x_norm_mean", np.zeros((1, D), dtype=np.float32)),
            x_norm_std=block_blobs[b].get("x_norm_std", np.ones((1, D), dtype=np.float32)),
        )
        model.import_parameter_blob(block_blobs[b], strict=True)
        t_b, W_b, _ = model.fetch_minibatch()
        X_pred, _, _ = model.predict(Xi_curr, t_b, W_b, const_value=1.0)
        Xi_curr = X_pred[:, -1, :].astype(np.float32)
        boundary_samples.append(Xi_curr.copy())
        model.sess.close()

    return boundary_samples


def run_recursive_training(
    Xi_generator,
    params,
    M,
    N_per_block,
    D,
    T_total,
    block_size,
    layers,
    stage_plan,
    final_plan,
    output_dir,
    precision_margin=0.10,
    max_refine_rounds=3,
    rollout_M=2000,
    save_tf_checkpoints=True,
    training_plan_rules: Optional[List[Dict]] = None,
    pass1_warm_start_from_next=False,
    cross_pass_warm_start: bool = True,
    n_passes: int = 2,
    resume_models_dir: str = "",
    resume_from_pass: int = 0,
    empirical_jitter_scale: float = 0.02,
    on_pass_end: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    if int(n_passes) < 1:
        raise ValueError("n_passes must be >= 1")
    if int(resume_from_pass) < 0:
        raise ValueError("resume_from_pass must be >= 0")

    blocks = build_blocks(T_total=T_total, block_size=block_size)
    print(
        f"[Recursive] blocks={len(blocks)} -> {[ (b['t_start'], b['t_end']) for b in blocks ]}, "
        f"n_passes={int(n_passes)}"
    )

    def _run_pass(
        pass_id,
        generators_per_block,
        warm_start_blobs=None,
        warm_start_from_next=False,
        prev_pass_loss_by_block=None,
    ):
        pass_dir = os.path.join(output_dir, f"pass_{pass_id}")
        os.makedirs(pass_dir, exist_ok=True)

        next_blob = None
        block_blobs = [None] * len(blocks)
        logs = []
        reference_loss = None

        for b in range(len(blocks) - 1, -1, -1):
            block = blocks[b]
            label = f"{_pass_label(pass_id)}:block{b}"
            print(
                f"\n[RecursiveBlock] {label} t=[{block['t_start']:.2f},{block['t_end']:.2f}] "
                f"T_block={block['T_block']:.2f}"
            )

            x_mean, x_std = estimate_generator_stats(generators_per_block[b], D=D, n_samples=max(4096, M))

            tf.reset_default_graph()
            model = NN_Quadratic_Coupled_Recursive(
                Xi_generator=generators_per_block[b],
                T=block["T_block"],
                M=M,
                N=N_per_block,
                D=D,
                layers=layers,
                parameters=params,
                t_start=block["t_start"],
                t_end=block["t_end"],
                T_total=T_total,
                terminal_blob=next_blob,
                normalize_time_input=True,
                x_norm_mean=x_mean,
                x_norm_std=x_std,
            )

            # Opzione: nella passata 1 inizializza il blocco i coi pesi del blocco i+1.
            if warm_start_from_next and next_blob is not None:
                model.import_parameter_blob(next_blob, strict=False)

            if warm_start_blobs is not None and warm_start_blobs[b] is not None:
                model.import_parameter_blob(warm_start_blobs[b], strict=False)

            precision_target = None
            if prev_pass_loss_by_block is not None and b in prev_pass_loss_by_block:
                precision_target = float(prev_pass_loss_by_block[b]) * (1.0 + precision_margin)
            elif reference_loss is not None:
                precision_target = reference_loss * (1.0 + precision_margin)

            default_refine_plan = [(50, 1e-5), (50, 5e-6)]
            resolved_plan = resolve_training_plan_for_block(
                rules=training_plan_rules or [],
                pass_id=pass_id,
                block_idx=b,
                n_blocks=len(blocks),
                default_stage=stage_plan,
                default_final=final_plan,
                default_refine=default_refine_plan,
            )

            block_stats = train_with_standard_schedule(
                model=model,
                stage_plan=resolved_plan["stage_plan"],
                final_plan=resolved_plan["final_plan"],
                eval_batches=5,
                precision_target=precision_target,
                max_refine_rounds=max_refine_rounds,
                refine_plan=resolved_plan["refine_plan"],
                label=label,
            )

            eval_loss = block_stats["eval_stats"]["mean_loss"]
            if reference_loss is None:
                reference_loss = eval_loss
                print(f"[Recursive] reference_loss set from terminal block: {reference_loss:.6e}")

            blob = model.export_parameter_blob()
            blob_path = os.path.join(pass_dir, f"block_{b:02d}.npz")
            save_blob_npz(blob, blob_path)
            ckpt_path = None
            if save_tf_checkpoints:
                ckpt_path = os.path.join(pass_dir, f"block_{b:02d}.ckpt")
                model.save_model(ckpt_path)

            log_row = {
                "pass": int(pass_id),
                "block": int(b),
                "t_start": float(block["t_start"]),
                "t_end": float(block["t_end"]),
                "T_block": float(block["T_block"]),
                "eval_mean_loss": float(block_stats["eval_stats"]["mean_loss"]),
                "eval_std_loss": float(block_stats["eval_stats"]["std_loss"]),
                "eval_mean_loss_per_sample": float(block_stats["eval_stats"]["mean_loss_per_sample"]),
                "eval_std_loss_per_sample": float(block_stats["eval_stats"]["std_loss_per_sample"]),
                "eval_mean_y0": float(block_stats["eval_stats"]["mean_y0"]),
                "precision_target": None
                if block_stats["precision_target"] is None
                else float(block_stats["precision_target"]),
                "refine_rounds": int(block_stats["refine_rounds"]),
                "stage_plan_used": resolved_plan["stage_plan"],
                "final_plan_used": resolved_plan["final_plan"],
                "refine_plan_used": resolved_plan["refine_plan"],
                "blob_path": blob_path,
                "ckpt_path": ckpt_path,
            }
            logs.append(log_row)

            block_blobs[b] = blob
            next_blob = blob

            model.sess.close()

        logs = sorted(logs, key=lambda x: x["block"])
        return block_blobs, logs, float(reference_loss), pass_dir

    pass_results = []
    prev_blobs = None
    prev_boundary_samples = None
    prev_pass_loss_by_block = None
    resumed_from = None
    start_pass_id = 1

    resume_models_dir = str(resume_models_dir or "").strip()
    if resume_models_dir != "":
        resume_models_dir = resolve_resume_models_dir(resume_models_dir)
        available_passes = detect_available_recursive_passes(resume_models_dir)
        if len(available_passes) == 0:
            raise FileNotFoundError(
                f"No pass_* directories found in resume_models_dir: {resume_models_dir}"
            )

        if int(resume_from_pass) > 0:
            loaded_pass_id = int(resume_from_pass)
            if loaded_pass_id not in available_passes:
                raise ValueError(
                    f"Requested resume_from_pass={loaded_pass_id} not found in "
                    f"{resume_models_dir}. Available: {available_passes}"
                )
        else:
            loaded_pass_id = int(max(available_passes))

        if loaded_pass_id >= int(n_passes):
            raise ValueError(
                f"Loaded pass={loaded_pass_id} but n_passes={int(n_passes)}. "
                "Set n_passes > loaded pass to continue training."
            )

        prev_blobs = load_pass_blobs_from_models_dir(
            models_dir=resume_models_dir,
            pass_id=loaded_pass_id,
            blocks=blocks,
            layers=layers,
            D=D,
            T_total=T_total,
        )
        prev_boundary_samples = rollout_boundaries(
            block_blobs=prev_blobs,
            blocks=blocks,
            Xi_generator=Xi_generator,
            params=params,
            M_rollout=rollout_M,
            N_per_block=N_per_block,
            D=D,
            layers=layers,
            T_total=T_total,
        )
        start_pass_id = loaded_pass_id + 1
        resumed_from = {
            "models_dir": resume_models_dir,
            "loaded_pass_id": int(loaded_pass_id),
            "available_passes": available_passes,
        }
        print(
            f"[Resume] loaded {_pass_label(loaded_pass_id)} from {resume_models_dir}, "
            f"continuing from {_pass_label(start_pass_id)}"
        )

    for pass_id in range(start_pass_id, int(n_passes) + 1):
        if pass_id == 1:
            generators = [Xi_generator for _ in blocks]
            warm_start = None
            warm_from_next = bool(pass1_warm_start_from_next)
        else:
            if prev_boundary_samples is None:
                if prev_blobs is None:
                    raise RuntimeError("Internal error: missing previous blobs for pass>=2")
                prev_boundary_samples = rollout_boundaries(
                    block_blobs=prev_blobs,
                    blocks=blocks,
                    Xi_generator=Xi_generator,
                    params=params,
                    M_rollout=rollout_M,
                    N_per_block=N_per_block,
                    D=D,
                    layers=layers,
                    T_total=T_total,
                )
            generators = [
                make_empirical_generator(prev_boundary_samples[b], jitter_scale=empirical_jitter_scale)
                for b in range(len(blocks))
            ]
            warm_start = prev_blobs if bool(cross_pass_warm_start) else None
            warm_from_next = False

        blobs_i, logs_i, ref_loss_i, pass_dir_i = _run_pass(
            pass_id=pass_id,
            generators_per_block=generators,
            warm_start_blobs=warm_start,
            warm_start_from_next=warm_from_next,
            prev_pass_loss_by_block=prev_pass_loss_by_block,
        )

        prev_blobs = blobs_i
        prev_pass_loss_by_block = {
            int(row["block"]): float(row["eval_mean_loss"])
            for row in logs_i
            if "eval_mean_loss" in row
        }
        prev_boundary_samples = rollout_boundaries(
            block_blobs=blobs_i,
            blocks=blocks,
            Xi_generator=Xi_generator,
            params=params,
            M_rollout=rollout_M,
            N_per_block=N_per_block,
            D=D,
            layers=layers,
            T_total=T_total,
        )

        pass_results.append(
            {
                "pass_id": int(pass_id),
                "reference_loss": float(ref_loss_i),
                "logs": logs_i,
                "blobs": blobs_i,
                "models_dir": pass_dir_i,
            }
        )

        if on_pass_end is not None:
            on_pass_end(
                {
                    "pass_id": int(pass_id),
                    "passes": list(pass_results),
                    "blocks": blocks,
                    "resumed_from": resumed_from,
                    "boundary_samples": prev_boundary_samples if prev_boundary_samples is not None else [],
                }
            )

    result = {
        "blocks": blocks,
        "passes": pass_results,
        "boundary_samples": prev_boundary_samples if prev_boundary_samples is not None else [],
        "resumed_from": resumed_from,
    }

    for item in pass_results:
        if item["pass_id"] == 1:
            result["pass1"] = {
                "logs": item["logs"],
                "reference_loss": item["reference_loss"],
                "blobs": item["blobs"],
            }
        if item["pass_id"] == 2:
            result["pass2"] = {
                "logs": item["logs"],
                "reference_loss": item["reference_loss"],
                "blobs": item["blobs"],
            }

    return result


###############################################################################
# Main comparabile: standard vs recursive
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Recursive time-stitching experiment (TF1)")
    parser.add_argument("--mode", type=str, default="recursive", choices=["standard", "recursive", "both"])
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--N", type=int, default=100, help="N steps per block")
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--T_standard", type=float, default=12.0)
    parser.add_argument("--T_total", type=float, default=48.0)
    parser.add_argument("--block_size", type=float, default=12.0)
    parser.add_argument("--output_dir", type=str, default="recursive1_outputs")
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Numero totale di pass ricorsive da eseguire (>=1).",
    )
    parser.add_argument(
        "--resume_models_dir",
        type=str,
        default="",
        help=(
            "Directory con pass_*/block_XX.npz di una run precedente "
            "da cui riprendere (es. .../recursive/models)."
        ),
    )
    parser.add_argument(
        "--resume_from_pass",
        type=int,
        default=0,
        help="Pass di partenza nel resume. 0=auto (massima disponibile in resume_models_dir).",
    )
    parser.add_argument(
        "--empirical_jitter_scale",
        type=float,
        default=0.02,
        help="Rumore relativo usato nel generatore empirico per pass >= 2.",
    )
    parser.add_argument(
        "--training_plan_csv",
        type=str,
        default="",
        help=(
            "CSV opzionale con piano training per blocco/pass. "
            "Colonne richieste: pass_scope,block_scope,phase,n_iter,lr "
            "(opzionali: order,enabled)."
        ),
    )
    parser.add_argument(
        "--pass1_warm_start_from_next",
        action="store_true",
        help=(
            "Se attivo, in pass1 il blocco i viene inizializzato coi pesi del blocco i+1 "
            "(quando disponibile). Le passate successive possono usare warm-start dal pass "
            "precedente (default attivo, disattivabile con --disable_cross_pass_warm_start)."
        ),
    )
    parser.add_argument(
        "--disable_cross_pass_warm_start",
        action="store_true",
        help=(
            "Se attivo, disabilita il warm-start dalle passate precedenti "
            "(warm_start=prev_blobs) per pass>=2."
        ),
    )
    parser.add_argument(
        "--exact_solution",
        type=str,
        default="none",
        help=(
            "Profilo opzionale per confronto con soluzione esatta. "
            "Valori supportati: none, quadratic_coupled"
        ),
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="auto",
        choices=["auto", "loss", "exact_mae_y", "exact_rmse_y", "exact_abs_y0"],
        help=(
            "Metrica di selezione della pass finale: "
            "auto usa exact_mae_y se exact_solution è attiva, altrimenti loss."
        ),
    )
    parser.add_argument(
        "--exact_regression_tolerance",
        type=float,
        default=0.20,
        help=(
            "Tolleranza regressione relativa tra pass consecutive su mean_abs_error_y "
            "(es. 0.20 = +20%). <=0 disabilita il guardrail."
        ),
    )
    parser.add_argument(
        "--exact_regression_action",
        type=str,
        default="warn",
        choices=["warn", "error", "ignore"],
        help="Azione quando il guardrail exact rileva regressione oltre soglia.",
    )
    parser.add_argument(
        "--eval_bundle_path",
        type=str,
        default="",
        help=(
            "Percorso opzionale a evaluation_bundle.npz da riusare per confronto path-by-path "
            "tra pass/run."
        ),
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=1234,
        help="Seed usato per costruire un evaluation bundle nuovo quando non viene caricato.",
    )
    parser.add_argument(
        "--allow_resume_without_plan",
        action="store_true",
        help=(
            "Permette resume con training_plan assente/non ereditabile. "
            "Di default il resume fallisce per evitare mismatch di schedule."
        ),
    )
    args = parser.parse_args()

    np.random.seed(1234)
    tf.set_random_seed(1234)

    M = args.M
    N = args.N
    D = args.D

    params = {
        "mu1": np.float32(1.0),
        "mu2": np.float32(1.0),
        "c1": np.float32(1.0),
        "c2": np.float32(1.0),
        "c3": np.float32(10.0),
        "c4": np.float32(10.0),
        "gamma": np.float32(1.0),
        "d": np.float32(1.0),
        "x_max": np.float32(10.0),
        "v_max": np.float32(2.0),
        "v_min": np.float32(-2.0),
        "s1": np.float32(0.5),
        "s2": np.float32(0.5),
        "s3": np.float32(0.5),
        "const": np.float32(0.0),
    }

    layers = [D + 1] + 4 * [256] + [1]
    stage_plan = [(5000, 1e-3), (5000, 5e-4), (5000, 1e-4), (5000, 5e-5)]
    final_plan = [(5000, 1e-5), (5000, 5e-6)]
    training_plan_rules = load_training_plan_csv(args.training_plan_csv)
    training_plan_effective_source = str(args.training_plan_csv or "").strip()
    training_plan_inherited_from_resume = False
    training_plan_inherited_run_config = None
    training_plan_inherited_csv = None

    resume_models_dir_arg = str(args.resume_models_dir or "").strip()
    resume_models_dir_resolved = (
        resolve_resume_models_dir(resume_models_dir_arg) if resume_models_dir_arg != "" else ""
    )

    if resume_models_dir_resolved != "" and len(training_plan_rules) == 0:
        inherited_rules, resume_cfg_path, resume_plan_csv = load_training_plan_rules_from_resume_run(
            resume_models_dir_resolved
        )
        if len(inherited_rules) > 0:
            training_plan_rules = inherited_rules
            training_plan_effective_source = f"inherited_from_resume:{resume_cfg_path}"
            training_plan_inherited_from_resume = True
            training_plan_inherited_run_config = resume_cfg_path
            training_plan_inherited_csv = resume_plan_csv
            print(
                f"[TrainingPlan] inherited {len(training_plan_rules)} rules from resume run config: "
                f"{resume_cfg_path}"
            )
        else:
            msg = (
                "Resume requested but no training plan was provided and no reusable "
                "training_plan_rules were found in the resumed run config. "
                "Pass --training_plan_csv (recommended) or --allow_resume_without_plan to proceed."
            )
            if bool(args.allow_resume_without_plan):
                print(f"[TrainingPlan][WARN] {msg}")
            else:
                raise ValueError(msg)

    if len(training_plan_rules) > 0:
        print(
            f"[TrainingPlan] loaded {len(training_plan_rules)} rules from {training_plan_effective_source}"
        )

    exact_solution = build_exact_solution_functions(
        solution_name=args.exact_solution,
        params=params,
        D=D,
    )
    if exact_solution is None:
        print("[ExactSolution] disabled")
    else:
        print(f"[ExactSolution] enabled profile='{exact_solution['name']}'")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_root, exist_ok=True)

    run_config = {
        "timestamp": run_id,
        "mode": args.mode,
        "M": M,
        "N": N,
        "D": D,
        "T_standard": args.T_standard,
        "T_total": args.T_total,
        "block_size": args.block_size,
        "passes": int(args.passes),
        "resume_models_dir": args.resume_models_dir,
        "resume_models_dir_resolved": resume_models_dir_resolved,
        "resume_from_pass": int(args.resume_from_pass),
        "empirical_jitter_scale": float(args.empirical_jitter_scale),
        "layers": layers,
        "stage_plan": stage_plan,
        "final_plan": final_plan,
        "training_plan_csv": args.training_plan_csv,
        "training_plan_effective_source": training_plan_effective_source,
        "training_plan_rules_count": len(training_plan_rules),
        "training_plan_rules": training_plan_rules,
        "training_plan_inherited_from_resume": bool(training_plan_inherited_from_resume),
        "training_plan_inherited_run_config": training_plan_inherited_run_config,
        "training_plan_inherited_csv": training_plan_inherited_csv,
        "pass1_warm_start_from_next": bool(args.pass1_warm_start_from_next),
        "cross_pass_warm_start": not bool(args.disable_cross_pass_warm_start),
        "selection_metric": str(args.selection_metric),
        "exact_regression_tolerance": float(args.exact_regression_tolerance),
        "exact_regression_action": str(args.exact_regression_action),
        "eval_bundle_path": str(args.eval_bundle_path),
        "eval_seed": int(args.eval_seed),
        "allow_resume_without_plan": bool(args.allow_resume_without_plan),
        "exact_solution": "none" if exact_solution is None else exact_solution["name"],
        "params": params,
        "plotting_available": _PLOTTING_AVAILABLE,
    }
    save_json(run_config, os.path.join(run_root, "run_config.json"))
    print(f"[Artifacts] run directory: {run_root}")

    if args.mode in ("standard", "both"):
        print("\n==================== STANDARD ====================")
        std_dir = os.path.join(run_root, "standard")
        os.makedirs(std_dir, exist_ok=True)
        model_std, logs_std = run_standard_reference(
            Xi_generator=Xi_generator_default,
            params=params,
            M=M,
            N=N,
            D=D,
            T=args.T_standard,
            layers=layers,
            stage_plan=stage_plan,
            final_plan=final_plan,
        )

        std_ckpt_path = os.path.join(std_dir, "model.ckpt")
        model_std.save_model(std_ckpt_path)

        std_blob = export_standard_parameter_blob(model_std)
        std_blob_path = os.path.join(std_dir, "model_weights.npz")
        save_blob_npz(std_blob, std_blob_path)

        save_rows_csv(logs_std.get("stage_logs", []), os.path.join(std_dir, "stage_logs.csv"))
        plot_stage_logs(
            logs_std.get("stage_logs", []),
            out_prefix=os.path.join(std_dir, "standard"),
            title="Standard",
        )

        std_summary = {
            "final_eval": logs_std.get("eval_stats", {}),
            "refine_rounds": logs_std.get("refine_rounds", 0),
            "checkpoint_path": std_ckpt_path,
            "weights_npz_path": std_blob_path,
        }
        if exact_solution is not None:
            t_test, W_test, Xi_test = model_std.fetch_minibatch()
            X_pred, Y_pred, Z_pred = model_std.predict(Xi_test, t_test, W_test, const_value=1.0)
            stitched_std = {
                "t": t_test.astype(np.float32),
                "X": X_pred.astype(np.float32),
                "Y": Y_pred.astype(np.float32),
                "Z": Z_pred.astype(np.float32),
            }
            exact_std = compute_stitched_exact_bundle(
                stitched=stitched_std,
                exact_solution=exact_solution,
            )
            print(
                "[Exact][Standard] "
                f"mean_pred_Y0={exact_std['summary']['mean_pred_y0']:.6f}, "
                f"mean_exact_Y0={exact_std['summary']['mean_exact_y0']:.6f}, "
                f"abs_err_Y0={exact_std['summary']['abs_error_mean_y0']:.6e}"
            )

            save_json(
                {
                    "summary": exact_std["summary"],
                    "timeseries": exact_std["timeseries"],
                },
                os.path.join(std_dir, "exact_metrics.json"),
            )
            save_exact_error_timeseries_csv(
                exact_std["timeseries"],
                os.path.join(std_dir, "exact_errors.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=stitched_std,
                Y_exact=exact_std["Y_exact"],
                Z_exact=exact_std["Z_exact"],
                blocks=[{"t_start": 0.0, "t_end": float(args.T_standard), "T_block": float(args.T_standard)}],
                out_dir=os.path.join(std_dir, "plots"),
                sample_paths=8,
                file_suffix="",
            )
            std_summary["exact_solution"] = {
                "enabled": True,
                "profile": exact_solution["name"],
                "summary": exact_std["summary"],
            }
        else:
            std_summary["exact_solution"] = {"enabled": False, "profile": "none"}
        save_json(std_summary, os.path.join(std_dir, "results.json"))

        print(f"[STANDARD] final eval: {logs_std['eval_stats']}")
        model_std.sess.close()

    if args.mode in ("recursive", "both"):
        print("\n==================== RECURSIVE ====================")
        rec_dir = os.path.join(run_root, "recursive")
        os.makedirs(rec_dir, exist_ok=True)

        explicit_eval_bundle = str(args.eval_bundle_path or "").strip()
        resume_eval_bundle = _find_resume_eval_bundle_path(resume_models_dir_resolved)
        if explicit_eval_bundle != "":
            eval_bundle_path = os.path.abspath(os.path.expanduser(explicit_eval_bundle))
        elif resume_eval_bundle is not None:
            eval_bundle_path = os.path.abspath(os.path.expanduser(resume_eval_bundle))
        else:
            eval_bundle_path = os.path.abspath(os.path.join(rec_dir, "evaluation_bundle.npz"))

        pass_plot_summary_holder = {"summary": None}

        def _on_recursive_pass_end(progress: Dict[str, Any]) -> None:
            passes_so_far = sorted(progress.get("passes", []), key=lambda x: int(x["pass_id"]))
            if len(passes_so_far) == 0:
                return
            pass_id = int(progress.get("pass_id", passes_so_far[-1]["pass_id"]))
            is_last_requested_pass = pass_id >= int(args.passes)
            print(
                f"\n[RecursivePlot] completed {_pass_label(pass_id)}: "
                f"updating cumulative plots up to {_pass_label(pass_id)}"
            )
            pass_plot_summary_holder["summary"] = print_recursive_pass(
                pass_entries=passes_so_far,
                blocks=progress.get("blocks", []),
                rec_dir=rec_dir,
                params=params,
                N_per_block=N,
                D=D,
                layers=layers,
                T_total=args.T_total,
                exact_solution=exact_solution,
                selection_metric=str(args.selection_metric),
                exact_regression_tolerance=float(args.exact_regression_tolerance),
                exact_regression_action=str(args.exact_regression_action),
                eval_bundle_path=eval_bundle_path,
                eval_seed=int(args.eval_seed),
                eval_min_paths=max(64, M),
                sample_paths=8,
                enforce_exact_regression_guardrail=is_last_requested_pass,
                print_compact_logs=is_last_requested_pass,
            )

        rec = run_recursive_training(
            Xi_generator=Xi_generator_default,
            params=params,
            M=M,
            N_per_block=N,
            D=D,
            T_total=args.T_total,
            block_size=args.block_size,
            layers=layers,
            stage_plan=stage_plan,
            final_plan=final_plan,
            output_dir=os.path.join(rec_dir, "models"),
            precision_margin=0.10,
            max_refine_rounds=3,
            rollout_M=max(2000, M),
            save_tf_checkpoints=True,
            training_plan_rules=training_plan_rules,
            pass1_warm_start_from_next=bool(args.pass1_warm_start_from_next),
            cross_pass_warm_start=not bool(args.disable_cross_pass_warm_start),
            n_passes=int(args.passes),
            resume_models_dir=resume_models_dir_resolved,
            resume_from_pass=int(args.resume_from_pass),
            empirical_jitter_scale=float(args.empirical_jitter_scale),
            on_pass_end=_on_recursive_pass_end,
        )

        pass_entries = sorted(rec.get("passes", []), key=lambda x: int(x["pass_id"]))
        if len(pass_entries) == 0:
            raise RuntimeError("No pass results available after recursive training")

        expected_pass_ids = sorted(int(p["pass_id"]) for p in pass_entries)
        plot_summary = pass_plot_summary_holder.get("summary", None)
        if plot_summary is None or plot_summary.get("processed_pass_ids", []) != expected_pass_ids:
            plot_summary = print_recursive_pass(
                pass_entries=pass_entries,
                blocks=rec["blocks"],
                rec_dir=rec_dir,
                params=params,
                N_per_block=N,
                D=D,
                layers=layers,
                T_total=args.T_total,
                exact_solution=exact_solution,
                selection_metric=str(args.selection_metric),
                exact_regression_tolerance=float(args.exact_regression_tolerance),
                exact_regression_action=str(args.exact_regression_action),
                eval_bundle_path=eval_bundle_path,
                eval_seed=int(args.eval_seed),
                eval_min_paths=max(64, M),
                sample_paths=8,
                enforce_exact_regression_guardrail=True,
                print_compact_logs=True,
            )

        exact_summary_by_pass = plot_summary["exact_summary_by_pass"]
        exact_summary_by_pass_index = plot_summary.get("exact_summary_by_pass_index", {})

        boundary_stats = []
        for i, arr in enumerate(rec.get("boundary_samples", [])):
            boundary_stats.append(
                {
                    "boundary_idx": int(i),
                    "n_samples": int(arr.shape[0]),
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                    "min": np.min(arr, axis=0),
                    "max": np.max(arr, axis=0),
                }
            )

        passes_summary = []
        for p in pass_entries:
            pass_id = int(p["pass_id"])
            passes_summary.append(
                {
                    "pass_id": pass_id,
                    "pass_index": _pass_index(pass_id),
                    "reference_loss": float(p["reference_loss"]),
                    "logs": p.get("logs", []),
                    "models_dir": p.get("models_dir", None),
                }
            )
        pass_summary_by_index = {int(p["pass_index"]): p for p in passes_summary}

        rec_summary = {
            "blocks": rec["blocks"],
            "passes": passes_summary,
            "resumed_from": rec.get("resumed_from", None),
            "boundary_stats": boundary_stats,
            "models_dir": os.path.join(rec_dir, "models"),
            "evaluation_bundle_path": plot_summary["eval_bundle_path"],
            "evaluation_bundle_M": int(plot_summary["evaluation_bundle_M"]),
            "selected_pass_id": int(plot_summary["selected_pass_id"]),
            "selected_pass_index": int(plot_summary["selected_pass_index"]),
            "selected_score_metric": plot_summary["selected_score_metric"],
            "selected_score": float(plot_summary["selected_score"]),
            "selected_scores_by_pass": plot_summary["selected_scores_by_pass"],
            "selected_scores_by_pass_index": plot_summary["selected_scores_by_pass_index"],
            "loss_score_metric": plot_summary["score_key"],
            "loss_pass_scores": {str(k): float(v) for k, v in plot_summary["pass_scores_loss"].items()},
            "loss_pass_scores_by_index": {
                str(k): float(v) for k, v in plot_summary["pass_scores_loss_by_index"].items()
            },
        }
        if exact_solution is None:
            rec_summary["exact_solution"] = {"enabled": False, "profile": "none"}
        else:
            rec_summary["exact_solution"] = {
                "enabled": True,
                "profile": exact_solution["name"],
                "by_pass": {str(k): v for k, v in exact_summary_by_pass.items()},
                "by_pass_index": exact_summary_by_pass_index,
                "selected_pass_summary": exact_summary_by_pass.get(
                    int(plot_summary["selected_pass_id"]),
                    None,
                ),
            }
        if 0 in pass_summary_by_index:
            rec_summary["pass0"] = {
                "reference_loss": pass_summary_by_index[0]["reference_loss"],
                "logs": pass_summary_by_index[0]["logs"],
            }
        if 1 in pass_summary_by_index:
            rec_summary["pass1"] = {
                "reference_loss": pass_summary_by_index[1]["reference_loss"],
                "logs": pass_summary_by_index[1]["logs"],
            }
        if 2 in pass_summary_by_index:
            rec_summary["pass2"] = {
                "reference_loss": pass_summary_by_index[2]["reference_loss"],
                "logs": pass_summary_by_index[2]["logs"],
            }
        save_json(rec_summary, os.path.join(rec_dir, "results.json"))


if __name__ == "__main__":
    main()
