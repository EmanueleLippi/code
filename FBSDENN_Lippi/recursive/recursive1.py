import os
import time
import argparse
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

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
            losses.append(float(loss_value))
            y0s.append(list(y_value[:, 0, 0]))

        return {
            "const": float(current_const),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
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


def plot_recursive_pass_logs(pass1_logs: List[Dict], pass2_logs: List[Dict], out_dir: str) -> None:
    if not _PLOTTING_AVAILABLE:
        print("[Plot] matplotlib non disponibile: skip plot_recursive_pass_logs")
        return
    if (pass1_logs is None or len(pass1_logs) == 0) and (pass2_logs is None or len(pass2_logs) == 0):
        return

    os.makedirs(out_dir, exist_ok=True)

    p1 = sorted(pass1_logs or [], key=lambda r: r["block"])
    p2 = sorted(pass2_logs or [], key=lambda r: r["block"])

    if len(p1) > 0:
        b1 = np.array([r["block"] for r in p1], dtype=np.int32)
        l1 = np.array([r["eval_mean_loss"] for r in p1], dtype=np.float64)
        y1 = np.array([r["eval_mean_y0"] for r in p1], dtype=np.float64)
    else:
        b1 = np.array([], dtype=np.int32)
        l1 = np.array([], dtype=np.float64)
        y1 = np.array([], dtype=np.float64)

    if len(p2) > 0:
        b2 = np.array([r["block"] for r in p2], dtype=np.int32)
        l2 = np.array([r["eval_mean_loss"] for r in p2], dtype=np.float64)
        y2 = np.array([r["eval_mean_y0"] for r in p2], dtype=np.float64)
    else:
        b2 = np.array([], dtype=np.int32)
        l2 = np.array([], dtype=np.float64)
        y2 = np.array([], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    if len(b1) > 0:
        plt.plot(b1, l1, "b-o", label="pass1 loss")
    if len(b2) > 0:
        plt.plot(b2, l2, "r-o", label="pass2 loss")
    plt.yscale("log")
    plt.title("Recursive blocks - Eval Mean Loss")
    plt.xlabel("Block index")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_blocks_eval_loss.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    if len(b1) > 0:
        plt.plot(b1, y1, "b-o", label="pass1 y0")
    if len(b2) > 0:
        plt.plot(b2, y2, "r-o", label="pass2 y0")
    plt.title("Recursive blocks - Eval Mean Y0")
    plt.xlabel("Block index")
    plt.ylabel("Mean Y0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recursive_blocks_eval_y0.png"), dpi=160)
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
            if row is None:
                continue
            pass_scope = str(row.get("pass_scope", "")).strip()
            block_scope = str(row.get("block_scope", "")).strip().lower()
            phase = str(row.get("phase", "")).strip().lower()
            if pass_scope == "" or block_scope == "" or phase == "":
                continue
            if phase not in ("stage", "final", "refine"):
                raise ValueError(
                    f"Invalid phase '{phase}' in {csv_path}:{i_row} (allowed: stage, final, refine)"
                )

            enabled_raw = str(row.get("enabled", "1")).strip().lower()
            enabled = enabled_raw not in ("0", "false", "no", "off", "")
            if not enabled:
                continue

            order_raw = str(row.get("order", "0")).strip()
            order = int(order_raw) if order_raw != "" else 0

            n_iter = int(str(row.get("n_iter", "")).strip())
            lr = float(str(row.get("lr", "")).strip())
            if n_iter <= 0:
                raise ValueError(f"n_iter must be > 0 in {csv_path}:{i_row}")
            if lr <= 0:
                raise ValueError(f"lr must be > 0 in {csv_path}:{i_row}")

            rules.append(
                {
                    "pass_scope": pass_scope,
                    "block_scope": block_scope,
                    "phase": phase,
                    "order": int(order),
                    "n_iter": int(n_iter),
                    "lr": float(lr),
                    "source_row": int(i_row),
                }
            )

    return rules


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
):
    blocks = build_blocks(T_total=T_total, block_size=block_size)
    print(f"[Recursive] blocks={len(blocks)} -> {[ (b['t_start'], b['t_end']) for b in blocks ]}")

    def _run_pass(pass_id, generators_per_block, warm_start_blobs=None):
        pass_dir = os.path.join(output_dir, f"pass_{pass_id}")
        os.makedirs(pass_dir, exist_ok=True)

        next_blob = None
        block_blobs = [None] * len(blocks)
        logs = []
        reference_loss = None

        for b in range(len(blocks) - 1, -1, -1):
            block = blocks[b]
            label = f"pass{pass_id}:block{b}"
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

            if warm_start_blobs is not None and warm_start_blobs[b] is not None:
                model.import_parameter_blob(warm_start_blobs[b], strict=False)

            precision_target = None
            if reference_loss is not None:
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
        return block_blobs, logs, float(reference_loss)

    # Pass 1: bootstrap (generator di partenza uguale per tutti i blocchi)
    generators_pass1 = [Xi_generator for _ in blocks]
    blobs_pass1, logs_pass1, ref_loss_pass1 = _run_pass(
        pass_id=1, generators_per_block=generators_pass1, warm_start_blobs=None
    )

    # Stima boundary empirici da rollout stitched
    boundary_samples = rollout_boundaries(
        block_blobs=blobs_pass1,
        blocks=blocks,
        Xi_generator=Xi_generator,
        params=params,
        M_rollout=rollout_M,
        N_per_block=N_per_block,
        D=D,
        layers=layers,
        T_total=T_total,
    )

    # Pass 2: retraining con generatori empirici per blocco
    generators_pass2 = []
    for b in range(len(blocks)):
        generators_pass2.append(make_empirical_generator(boundary_samples[b], jitter_scale=0.02))

    blobs_pass2, logs_pass2, ref_loss_pass2 = _run_pass(
        pass_id=2, generators_per_block=generators_pass2, warm_start_blobs=blobs_pass1
    )

    return {
        "blocks": blocks,
        "pass1": {"logs": logs_pass1, "reference_loss": ref_loss_pass1, "blobs": blobs_pass1},
        "pass2": {"logs": logs_pass2, "reference_loss": ref_loss_pass2, "blobs": blobs_pass2},
        "boundary_samples": boundary_samples,
    }


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
        "--training_plan_csv",
        type=str,
        default="",
        help=(
            "CSV opzionale con piano training per blocco/pass. "
            "Colonne richieste: pass_scope,block_scope,phase,n_iter,lr "
            "(opzionali: order,enabled)."
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
    if len(training_plan_rules) > 0:
        print(
            f"[TrainingPlan] loaded {len(training_plan_rules)} rules from {args.training_plan_csv}"
        )

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
        "layers": layers,
        "stage_plan": stage_plan,
        "final_plan": final_plan,
        "training_plan_csv": args.training_plan_csv,
        "training_plan_rules_count": len(training_plan_rules),
        "training_plan_rules": training_plan_rules,
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
        save_json(std_summary, os.path.join(std_dir, "results.json"))

        print(f"[STANDARD] final eval: {logs_std['eval_stats']}")
        model_std.sess.close()

    if args.mode in ("recursive", "both"):
        print("\n==================== RECURSIVE ====================")
        rec_dir = os.path.join(run_root, "recursive")
        os.makedirs(rec_dir, exist_ok=True)
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
        )

        print("\n=== Recursive Log pass2 (compact) ===")
        for row in rec["pass2"]["logs"]:
            print(
                f"block={row['block']}, t=[{row['t_start']:.1f},{row['t_end']:.1f}], "
                f"eval_loss={row['eval_mean_loss']:.3e}, eval_y0={row['eval_mean_y0']:.3f}, "
                f"target={row['precision_target']}, refine={row['refine_rounds']}"
            )

        pass1_logs = rec["pass1"]["logs"]
        pass2_logs = rec["pass2"]["logs"]
        save_rows_csv(pass1_logs, os.path.join(rec_dir, "pass1_logs.csv"))
        save_rows_csv(pass2_logs, os.path.join(rec_dir, "pass2_logs.csv"))
        plot_recursive_pass_logs(pass1_logs, pass2_logs, os.path.join(rec_dir, "plots"))

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

        rec_summary = {
            "blocks": rec["blocks"],
            "pass1": {
                "reference_loss": rec["pass1"]["reference_loss"],
                "logs": pass1_logs,
            },
            "pass2": {
                "reference_loss": rec["pass2"]["reference_loss"],
                "logs": pass2_logs,
            },
            "boundary_stats": boundary_stats,
            "models_dir": os.path.join(rec_dir, "models"),
        }
        save_json(rec_summary, os.path.join(rec_dir, "results.json"))


if __name__ == "__main__":
    main()
