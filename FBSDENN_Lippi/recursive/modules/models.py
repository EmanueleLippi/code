import os
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from utils import _as_blob_dict, save_blob_npz


class FBSNN(tf.keras.Model, ABC):
    def __init__(
        self,
        Xi_generator,
        T,
        M,
        N,
        D,
        layers_dims,
        clip_grad_norm=1.0,
        use_antithetic_sampling=True,
    ):
        super(FBSNN, self).__init__()
        self.Xi_generator = Xi_generator
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.layers_dims = layers_dims
        self.clip_grad_norm = clip_grad_norm
        self.use_antithetic_sampling = bool(use_antithetic_sampling)

        # Build Neural Network architecture
        self.net = tf.keras.Sequential()
        initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg', distribution='truncated_normal'
        )
        for units in layers_dims[1:-1]:
            self.net.add(tf.keras.layers.Dense(units, activation=tf.math.sin, kernel_initializer=initializer))
        self.net.add(tf.keras.layers.Dense(layers_dims[-1], activation=None, kernel_initializer=initializer))

        self.const_val = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam()
        
        # Inizializza i pesi chiamando il modello su un tensore dummy
        dummy_t = tf.zeros((1, 1), dtype=tf.float32)
        dummy_x = tf.zeros((1, D), dtype=tf.float32)
        self.net(tf.concat([dummy_t, dummy_x], axis=1))

    def save_model(self, path: str) -> None:
        self.save_weights(path)

    def load_model(self, path: str) -> None:
        self.load_weights(path)

    @tf.function
    def net_u(self, t, X):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(X)
            u = self.net(tf.concat([t, X], axis=1))
        Du = tape.gradient(u, X)
        return u, Du

    @tf.function
    def Dg_tf(self, X):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(X)
            g = self.g_tf(X)
        return tape.gradient(g, X)

    @tf.function
    def loss_function(self, t, W, Xi):
        loss = tf.constant(0.0, dtype=tf.float32)

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi
        
        Y0, Du0 = self.net_u(t0, X0)
        sigma0 = self.sigma_tf(t0, X0, Y0)
        Z0 = tf.squeeze(tf.matmul(tf.expand_dims(Du0, 1), sigma0), axis=1)

        X_list = tf.TensorArray(tf.float32, size=self.N + 1)
        Y_list = tf.TensorArray(tf.float32, size=self.N + 1)
        Z_list = tf.TensorArray(tf.float32, size=self.N + 1)

        X_list = X_list.write(0, X0)
        Y_list = Y_list.write(0, Y0)
        Z_list = Z_list.write(0, Z0)

        for n in tf.range(self.N):
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

            X_list = X_list.write(n + 1, X0)
            Y_list = Y_list.write(n + 1, Y0)
            Z_list = Z_list.write(n + 1, Z0)

        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))

        Dg = self.Dg_tf(X1)
        Z_terminal = tf.squeeze(tf.matmul(tf.expand_dims(Dg, 1), sigma1), axis=1)
        loss += tf.reduce_sum(tf.square(Z1 - Z_terminal))

        X_stack = tf.transpose(X_list.stack(), [1, 0, 2])
        Y_stack = tf.transpose(Y_list.stack(), [1, 0, 2])
        Z_stack = tf.transpose(Z_list.stack(), [1, 0, 2])

        return loss / tf.cast(self.N, tf.float32), X_stack, Y_stack, Z_stack

    def fetch_minibatch(self):
        M, N, D = self.M, self.N, self.D
        Dt = np.zeros((M, N + 1, 1), dtype=np.float32)
        DW = np.zeros((M, N + 1, D), dtype=np.float32)
        dt = float(self.T) / N

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

        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        Xi_batch = self.Xi_generator(M, D).astype(np.float32)
        return tf.constant(t, dtype=tf.float32), tf.constant(W, dtype=tf.float32), tf.constant(Xi_batch, dtype=tf.float32)

    @tf.function
    def train_step(self, t, W, Xi, learning_rate):
        self.optimizer.learning_rate.assign(learning_rate)
        with tf.GradientTape() as tape:
            loss, X_pred, Y_pred, Z_pred = self.loss_function(t, W, Xi)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        if self.clip_grad_norm is not None and grads_and_vars:
            grads, vars_ = zip(*grads_and_vars)
            clipped, _ = tf.clip_by_global_norm(list(grads), self.clip_grad_norm)
            grads_and_vars = list(zip(clipped, vars_))

        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)
        return loss, X_pred, Y_pred, Z_pred

    def evaluate_model(self, const_value=None, n_batches=5):
        if const_value is not None:
            self.const_val.assign(const_value)
        
        losses, losses_per_sample, y0s = [], [], []
        for _ in range(n_batches):
            t_batch, W_batch, Xi_batch = self.fetch_minibatch()
            loss, X_pred, Y_pred, Z_pred = self.loss_function(t_batch, W_batch, Xi_batch)
            loss_value = float(loss)
            losses.append(loss_value)
            losses_per_sample.append(loss_value / float(self.M))
            y0s.append(list(Y_pred.numpy()[:, 0, 0]))

        return {
            "const": float(self.const_val.numpy()),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "mean_loss_per_sample": float(np.mean(losses_per_sample)),
            "std_loss_per_sample": float(np.std(losses_per_sample)),
            "mean_y0": float(np.mean(y0s)),
            "std_y0": float(np.std(y0s)),
            "n_batches": int(n_batches),
        }

    def predict_model(self, Xi_star, t_star, W_star, const_value=None):
        if const_value is not None:
            self.const_val.assign(const_value)
        _, X_star, Y_star, Z_star = self.loss_function(
            tf.convert_to_tensor(t_star, dtype=tf.float32),
            tf.convert_to_tensor(W_star, dtype=tf.float32),
            tf.convert_to_tensor(Xi_star, dtype=tf.float32),
        )
        return X_star.numpy(), Y_star.numpy(), Z_star.numpy()

    def clone_weights_from(self, weights_list):
        self.set_weights(weights_list)

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        return tf.zeros([tf.shape(X)[0], self.D], dtype=tf.float32)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        return tf.linalg.diag(tf.ones([tf.shape(X)[0], self.D], dtype=tf.float32))


def _init_quadratic_coupled_parameters(model, parameters):
    model.mu1 = tf.constant(parameters.mu1, dtype=tf.float32)
    model.mu2 = tf.constant(parameters.mu2, dtype=tf.float32)
    model.c1 = tf.constant(parameters.c1, dtype=tf.float32)
    model.c2 = tf.constant(parameters.c2, dtype=tf.float32)
    model.c3 = tf.constant(parameters.c3, dtype=tf.float32)
    model.c4 = tf.constant(parameters.c4, dtype=tf.float32)
    model.gamma = tf.constant(parameters.gamma, dtype=tf.float32)
    model.s1 = tf.constant(parameters.s1, dtype=tf.float32)
    model.s2 = tf.constant(parameters.s2, dtype=tf.float32)
    model.s3 = tf.constant(parameters.s3, dtype=tf.float32)
    model.x_max = tf.constant(parameters.x_max, dtype=tf.float32)
    model.v_min = tf.constant(parameters.v_min, dtype=tf.float32)
    model.v_max = tf.constant(parameters.v_max, dtype=tf.float32)
    model.d = tf.constant(parameters.d, dtype=tf.float32)
    if "const" in parameters.__dict__:
        model.const_val.assign(parameters.const)


class NN_Quadratic_Coupled(FBSNN):
    def __init__(
        self,
        Xi,
        T,
        M,
        N,
        D,
        layers,
        parameters,
        clip_grad_norm=1.0,
        use_antithetic_sampling=True,
    ):
        super().__init__(
            Xi,
            T,
            M,
            N,
            D,
            layers,
            clip_grad_norm=clip_grad_norm,
            use_antithetic_sampling=use_antithetic_sampling,
        )
        _init_quadratic_coupled_parameters(self, parameters)

    def psi(self, X_state):
        return tf.maximum(0.0, tf.minimum(1.0, tf.minimum(X_state / self.d, (self.x_max - X_state) / self.d)))

    def psi3(self, V):
        return tf.maximum(0.0, tf.minimum(1.0, (self.v_max - V) / self.d))

    def psi4(self, V):
        return tf.maximum(0.0, tf.minimum(1.0, (V - self.v_min) / self.d))

    def f(self, X, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        Z_S, Z_H, Z_V, _ = tf.split(Z, num_or_size_splits=4, axis=1)
        exp_S = tf.exp(-S)
        return -0.5 * V * self.psi(-exp_S * Z_S / (self.gamma * self.s1))

    def mu_tf(self, t, X, Y, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        dS = self.mu1 * (self.c1 - S)
        dH = self.mu2 * (self.c2 - H)
        dV = (
            self.f(X, Z) * self.psi(X_state)
            + self.c3 * self.psi(-X_state) * self.psi3(V)
            - self.c4 * self.psi(X_state - self.x_max) * self.psi4(V)
        )
        dX = V
        return tf.concat([dS, dH, dV, dX], axis=1)

    def g_tf(self, X):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        exp_S = tf.exp(S)
        return -self.gamma * exp_S * X_state + V ** 2 + V * X_state

    def phi_tf(self, t, X, Y, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        Z_S, Z_H, Z_V, _ = tf.split(Z, num_or_size_splits=4, axis=1)

        exp_S = tf.exp(S)
        term1 = -self.gamma * exp_S * X_state * self.mu1 * (self.c1 - S)
        term2 = (2 * V + X_state) * (
            self.f(X, Z) * self.psi(X_state)
            + self.c3 * self.psi(-X_state) * self.psi3(V)
            - self.c4 * self.psi(X_state - self.x_max) * self.psi4(V)
        )
        term3 = -self.gamma * exp_S * V + (0.5 * (Z_V / self.s3 - X_state)) ** 2
        term4 = -0.5 * self.gamma * exp_S * X_state * self.s1 ** 2 + self.s3 ** 2

        return term1 + term2 + term3 + term4

    def sigma_tf(self, t, X, Y):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        zeros = tf.zeros_like(S)
        ones = tf.ones_like(S)

        r1 = tf.concat([self.s1 * ones, zeros, zeros, zeros], axis=1)
        r2 = tf.concat([zeros, self.s2 * ones, zeros, zeros], axis=1)
        r3 = tf.concat([zeros, zeros, self.s3 * ones, zeros], axis=1)
        r4 = tf.concat([zeros, zeros, zeros, zeros], axis=1)

        return tf.stack([r1, r2, r3, r4], axis=1)


class FBSNN_Recursive(FBSNN):
    def __init__(
        self,
        Xi_generator,
        T,
        M,
        N,
        D,
        layers,
        t_start,
        t_end,
        T_total,
        terminal_blob=None,
        normalize_time_input=True,
        x_norm_mean=None,
        x_norm_std=None,
        clip_grad_norm=1.0,
        use_antithetic_sampling=True,
    ):
        self.t_start = np.float32(t_start)
        self.t_end = np.float32(t_end)
        self.T_total_val = np.float32(T_total)
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

        self._x_norm_mean_tf = tf.constant(self.x_norm_mean_np, dtype=tf.float32)
        self._x_norm_std_tf = tf.constant(self.x_norm_std_np, dtype=tf.float32)
        self._T_total_tf = tf.constant(self.T_total_val, dtype=tf.float32)

        # Call the base class explicitly to avoid the diamond-inheritance MRO
        # from routing through NN_Quadratic_Coupled.__init__.
        FBSNN.__init__(
            self,
            Xi_generator,
            T,
            M,
            N,
            D,
            layers,
            clip_grad_norm=clip_grad_norm,
            use_antithetic_sampling=use_antithetic_sampling,
        )

    def _normalize_t(self, t):
        if not self.normalize_time_input:
            return t
        return 2.0 * (t / self._T_total_tf) - 1.0

    def _normalize_x(self, X):
        return (X - self._x_norm_mean_tf) / self._x_norm_std_tf

    @tf.function
    def net_u(self, t, X):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(X)
            t_in = self._normalize_t(t)
            X_in = self._normalize_x(X)
            u = self.net(tf.concat([t_in, X_in], axis=1))
        Du = tape.gradient(u, X)
        return u, Du

    def fetch_minibatch(self):
        M, N, D = self.M, self.N, self.D
        Dt = np.zeros((M, N + 1, 1), dtype=np.float32)
        DW = np.zeros((M, N + 1, D), dtype=np.float32)
        dt = float(self.T) / N

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
        return tf.constant(t, dtype=tf.float32), tf.constant(W, dtype=tf.float32), tf.constant(Xi_batch, dtype=tf.float32)

    def _build_terminal_constants_if_needed(self):
        if self.terminal_blob is None:
            return
        if self._terminal_weights_tf is not None:
            return

        n_layers = int(self.terminal_blob["n_layers"])
        self._terminal_weights_tf = []
        self._terminal_biases_tf = []
        for i in range(n_layers):
            self._terminal_weights_tf.append(tf.constant(self.terminal_blob[f"W_{i}"], dtype=tf.float32))
            self._terminal_biases_tf.append(tf.constant(self.terminal_blob[f"b_{i}"], dtype=tf.float32))

        self._terminal_x_mean_tf = tf.constant(
            self.terminal_blob.get("x_norm_mean", np.zeros((1, self.D), dtype=np.float32)), dtype=tf.float32
        )
        self._terminal_x_std_tf = tf.constant(
            np.maximum(self.terminal_blob.get("x_norm_std", np.ones((1, self.D), dtype=np.float32)), 1.0e-3),
            dtype=tf.float32,
        )
        self._terminal_T_total_tf = tf.constant(
            np.float32(self.terminal_blob.get("T_total", self.T_total_val)), dtype=tf.float32
        )
        self._terminal_use_time = bool(int(self.terminal_blob.get("normalize_time_input", 1)))

    @tf.function
    def _terminal_u(self, t_abs, X):
        # Aggiunta manuale dei tensori layer-by-layer
        t_in = t_abs
        if self._terminal_use_time:
            t_in = 2.0 * (t_abs / self._terminal_T_total_tf) - 1.0
        X_in = (X - self._terminal_x_mean_tf) / self._terminal_x_std_tf
        
        H = tf.concat([t_in, X_in], axis=1)
        num_layers = len(self._terminal_weights_tf) + 1
        for l in range(0, num_layers - 2):
            W = self._terminal_weights_tf[l]
            b = self._terminal_biases_tf[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = self._terminal_weights_tf[-1]
        b = self._terminal_biases_tf[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    @abstractmethod
    def _model_g_tf(self, X):
        pass

    def g_tf(self, X):
        if self.terminal_blob is None:
            return self._model_g_tf(X)
        self._build_terminal_constants_if_needed()
        t_eval = tf.ones([tf.shape(X)[0], 1], dtype=tf.float32) * tf.constant(self.t_end, dtype=tf.float32)
        return self._terminal_u(t_eval, X)

    def Dg_tf(self, X):
        if self.terminal_blob is None:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(X)
                g = self._model_g_tf(X)
            return tape.gradient(g, X)
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(X)
            g = self.g_tf(X)
        return tape.gradient(g, X)

    def export_parameter_blob(self) -> Dict[str, np.ndarray]:
        # Keras models have multiple nested weights arrays
        values = self.net.get_weights()
        n_layers = len(values) // 2
        blob = {
            "n_layers": np.array(n_layers, dtype=np.int32),
            "layers": np.asarray(self.layers_dims, dtype=np.int32),
            "t_start": np.asarray(self.t_start, dtype=np.float32),
            "t_end": np.asarray(self.t_end, dtype=np.float32),
            "T_total": np.asarray(self.T_total_val, dtype=np.float32),
            "normalize_time_input": np.asarray(int(self.normalize_time_input), dtype=np.int32),
            "x_norm_mean": np.asarray(self.x_norm_mean_np, dtype=np.float32),
            "x_norm_std": np.asarray(self.x_norm_std_np, dtype=np.float32),
        }
        for i in range(n_layers):
            blob[f"W_{i}"] = values[i * 2].astype(np.float32)
            blob[f"b_{i}"] = values[i * 2 + 1].astype(np.float32)
        return blob

    def import_parameter_blob(self, blob_or_path, strict=True):
        blob = _as_blob_dict(blob_or_path)
        if blob is None:
            return

        n_layers = len(self.net.get_weights()) // 2
        if strict and int(blob["n_layers"]) != n_layers:
            raise ValueError(
                f"n_layers mismatch: model={n_layers}, blob={int(blob['n_layers'])}"
            )

        new_weights = []
        for i in range(n_layers):
            w_key = f"W_{i}"
            b_key = f"b_{i}"
            if w_key in blob:
                new_weights.append(blob[w_key])
            elif strict:
                raise KeyError(f"Missing key {w_key} in blob")

            if b_key in blob:
                new_weights.append(blob[b_key])
            elif strict:
                raise KeyError(f"Missing key {b_key} in blob")
        if len(new_weights) > 0:
            self.net.set_weights(new_weights)

    def save_parameter_blob(self, path: str) -> None:
        save_blob_npz(self.export_parameter_blob(), path)

    def load_parameter_blob(self, path: str, strict=True) -> None:
        self.import_parameter_blob(path, strict=strict)


class NN_Quadratic_Coupled_Recursive(FBSNN_Recursive, NN_Quadratic_Coupled):
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
        clip_grad_norm=1.0,
        use_antithetic_sampling=True,
    ):
        FBSNN_Recursive.__init__(
            self,
            Xi_generator=Xi_generator,
            T=T,
            M=M,
            N=N,
            D=D,
            layers=layers,
            t_start=t_start,
            t_end=t_end,
            T_total=T_total,
            terminal_blob=terminal_blob,
            normalize_time_input=normalize_time_input,
            x_norm_mean=x_norm_mean,
            x_norm_std=x_norm_std,
            clip_grad_norm=clip_grad_norm,
            use_antithetic_sampling=use_antithetic_sampling,
        )
        _init_quadratic_coupled_parameters(self, parameters)

    def _model_g_tf(self, X):
        return NN_Quadratic_Coupled.g_tf(self, X)
