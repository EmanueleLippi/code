## Rete neurale con normalizzazione della loss

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class FBSNN(ABC): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi_generator, T,
                       M, N, D,
                       layers,
                       clip_grad_norm=1.0, 
                       use_antithetic_sampling=True):
        self.Xi_generator = Xi_generator #initial data generator
        self.T = T # terminal time

        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions

        # layers
        self.layers = layers # (D+1) --> 1

        # migliorie advanced richieste
        self.clip_grad_norm = clip_grad_norm
        self.use_antithetic_sampling = bool(use_antithetic_sampling)

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # tf placeholders and graph (training)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[M, self.N+1, 1]) # M x (N+1) x 1
        self.W_tf = tf.placeholder(tf.float32, shape=[M, self.N+1, self.D]) # M x (N+1) x D
        self.Xi_tf = tf.placeholder(tf.float32, shape=[M, D]) # M x D
        self.const_tf = tf.placeholder(tf.float32, shape=[])

        self.loss, self.X_pred, self.Y_pred, self.Z_pred = self.loss_function(self.t_tf, self.W_tf, self.Xi_tf)

        # optimizer + gradient clipping
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

        # initialize session and variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Saver for saving/restoring variables
        self.saver = tf.train.Saver()

        # lightweight in-memory checkpoint utilities for early stopping
        self._trainable_vars = tf.trainable_variables()
        self._snapshot_ph = [
            tf.placeholder(tf.float32, shape=v.shape, name=f"snapshot_ph_{i}")
            for i, v in enumerate(self._trainable_vars)
        ]
        self._restore_ops = [v.assign(ph) for v, ph in zip(self._trainable_vars, self._snapshot_ph)]

    def save_model(self, path):
        """
        Save the trained model weights to disk.
        """
        save_path = self.saver.save(self.sess, path)
        print(f"Model saved in path: {save_path}")

    def load_model(self, path):
        """
        Restore model weights from disk.
        """
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

    def net_u(self, t, X): # M x 1, M x D
        u = self.neural_net(tf.concat([t, X], 1), self.weights, self.biases) # M x 1
        Du = tf.gradients(u, X)[0] # M x D
        return u, Du

    def Dg_tf(self, X): # M x D
        return tf.gradients(self.g_tf(X), X)[0] # M x D

    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = 0
        X_list = []
        Y_list = []
        Z_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        #X0 = tf.tile(Xi, [self.M, 1]) # M x D
        X0 = Xi  # già M x D
        Y0, Du0 = self.net_u(t0, X0) # M x 1, M x D
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

            # Z = Du*sigma e nella backward compare Z*dW.
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + tf.reduce_sum(Z0 * dW, axis=1, keepdims=True)

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

        return loss/self.N, X, Y, Z

    def fetch_minibatch(self):
        T = self.T

        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1)) # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D)) # M x (N+1) x D

        dt = T / N

        Dt[:, 1:, :] = dt
        if self.use_antithetic_sampling and M > 1:
            half_M = M // 2
            DW_half = np.sqrt(dt) * np.random.normal(size=(half_M, N, D))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M:2 * half_M, 1:, :] = -DW_half
            if M % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * np.random.normal(size=(N, D))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1) # M x (N+1) x 1
        W = np.cumsum(DW, axis=1) # M x (N+1) x D

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
        early_stopping_metric='loss',
        #y0_target=None,
        patience=None,
        min_delta=1e-3,
        restore_best=False,
    ):
        start_time = time.time()
        last_loss = None
        #last_y0 = None

        current_const = np.float32(self.const if const_value is None else const_value)

        best_score = np.inf
        best_iter = -1
        best_snapshot = None
        no_improve_iters = 0
        stopped_early = False

        for it in range(N_Iter):
            t_batch, W_batch, Xi_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D, M x D

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
                elapsed = time.time() - start_time
                loss_value, Y_value, learning_rate_value = self.sess.run([self.loss, self.Y_pred, self.learning_rate], tf_dict)
                last_loss = float(loss_value)
                mean_Y0 = np.mean(Y_value[:, 0, 0])
                print('It: %d, Loss: %.3e, Mean Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss_value, mean_Y0, elapsed, learning_rate_value))
                start_time = time.time()

            if (it % eval_every == 0) or (it == N_Iter - 1):
                eval_stats = self.evaluate(const_value=current_const, n_batches=val_batches)

                #if early_stopping_metric == 'y0_abs':
                #    if y0_target is None:
                #        raise ValueError("y0_target is required when early_stopping_metric='y0_abs'")
                #    score = abs(eval_stats['mean_y0'] - float(y0_target))
                if early_stopping_metric == 'loss':
                    score = eval_stats['mean_loss']

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
            'const': float(current_const),
            'learning_rate': float(learning_rate),
            'n_iter': int(N_Iter),
            'last_loss': last_loss,
            'best_iter': int(best_iter),
            'best_score': float(best_score),
            'stopped_early': bool(stopped_early),
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
            y0s.append(list(y_value[:,0,0]))

        return {
            'const': float(current_const),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'mean_y0': float(np.mean(y0s)),
            'std_y0': float(np.std(y0s)),
            'n_batches': int(n_batches),
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

    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x1

    @abstractmethod
    def g_tf(self, X): # M x D
        pass # M x 1

    @abstractmethod
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return np.zeros([M, D]) # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.matrix_diag(tf.ones([M, D])) # M x D x D
    ###########################################################################

class NN_Quadratic_Coupled(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, parameters):
        self.mu1 = parameters['mu1']
        self.mu2 = parameters['mu2']
        self.c1 = parameters['c1']
        self.c2 = parameters['c2']
        self.c3 = parameters['c3']
        self.c4 = parameters['c4']
        self.gamma = parameters['gamma']
        self.s1 = parameters['s1']
        self.s2 = parameters['s2']
        self.s3 = parameters['s3']
        self.x_max = parameters['x_max']
        self.v_min = parameters['v_min']
        self.v_max = parameters['v_max']
        self.d = parameters['d']
        self.const = parameters['const']
        super().__init__(Xi, T, M, N, D, layers)

    def psi(self,X_state):
        #S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        result = tf.maximum(0.0,
                   tf.minimum(
                     1.0,
                     tf.minimum(X_state/self.d, (self.x_max - X_state)/self.d)
                     )
                 )
        return result

    def psi3(self,V):
        #S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        result = tf.maximum(0.0,
                   tf.minimum(
                     1.0,
                     (self.v_max- V)/self.d
                     )
                 )
        return result

    def psi4(self,V):
        #S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        result = tf.maximum(0.0,
                   tf.minimum(
                     1.0,
                     (V - self.v_min)/self.d
                     )
                 )
        return result

    def f(self,X,Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        Z_S, Z_H, Z_V, _ = tf.split(Z, num_or_size_splits=4, axis=1)
        s1 = tf.cast(self.s1, tf.float32)
        gamma = tf.cast(self.gamma, tf.float32)
        exp_S = tf.exp(-S)
        return -0.5 * V * self.psi(- exp_S * Z_S /(gamma * s1))

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
        dV = self.f(X,Z) * self.psi(X_state) + c3 * self.psi(-X_state)*self.psi3(V) - c4 * self.psi(X_state-x_max)*self.psi4(V)
        dX = V
        return tf.concat([dS, dH, dV, dX], axis=1)

    def g_tf(self, X):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        gamma = tf.cast(self.gamma, tf.float32)
        exp_S = tf.exp(S)
        return -gamma * exp_S * X_state + V**2 + V*X_state

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

        term1 = -gamma * exp_S * X_state * mu1 * (c1-S)
        term2 = (2 * V + X_state) * (self.f(X,Z) * self.psi(X_state) + c3 * self.psi(-X_state)*self.psi3(V) - c4 * self.psi(X_state-x_max)*self.psi4(V))
        term3 = -gamma * exp_S * V + (0.5 * (Z_V/s3 - X_state))**2
        # V = 0.5 * (Z_V/s3 - X_state)
        term4 = -0.5 * gamma * exp_S * X_state * s1**2 + s3**2

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



def train_plan(model_class, M, N, D, T, Xi_generator, params, layers, stage_plan, final_plan, curriculum_mode=False, save_path=None, log_path=None, plot_path=None, u_exact=None, z_exact=None):
    tf.reset_default_graph()
    model = model_class(Xi_generator, T, M, N, D, layers, params)
    stage_logs = []
    if curriculum_mode:
        # Curriculum sull'accoppiamento: const cresce gradualmente fino a 1.0
        coupling_step = 0.2
        coupling_levels = np.arange(0.0, 1.0 + coupling_step/2, coupling_step, dtype=np.float32)
    else:
        coupling_levels = [1.0]
    

    for level in coupling_levels:
        model.const = np.float32(level)
        print(f"=== Coupling stage: const={float(level):.1f} ===")
        for n_iter, lr in stage_plan:
            t0 = time.time()
            train_stats = model.train(N_Iter=n_iter, learning_rate=lr, const_value=level)
            eval_stats = model.evaluate(const_value=level, n_batches=5)
            elapsed = time.time() - t0
            log_row = {
                'phase': 'curriculum',
                'const': float(level),
                'lr': float(lr),
                'n_iter': int(n_iter),
                'train_last_loss': train_stats['last_loss'],
                'eval_mean_loss': eval_stats['mean_loss'],
                'eval_std_loss': eval_stats['std_loss'],
                'eval_mean_y0': eval_stats['mean_y0'],
                'eval_std_y0': eval_stats['std_y0'],
                'elapsed_sec': float(elapsed),
            }
            stage_logs.append(log_row)
            print(
                f"[StageSummary] const={level:.1f}, lr={lr:.1e}, iters={n_iter}, "
                f"eval_loss={eval_stats['mean_loss']:.3e}±{eval_stats['std_loss']:.2e}, "
                f"eval_Y0={eval_stats['mean_y0']:.3f}±{eval_stats['std_y0']:.3f}, "
                f"time={elapsed:.1f}s"
            )

    # Fine-tuning finale fully coupled
    model.const = np.float32(1.0)
    print("=== Final fine-tuning ===")

    for n_iter, lr in final_plan:
        t0 = time.time()
        train_stats = model.train(N_Iter=n_iter, learning_rate=lr, const_value=1.0,
                                        eval_every=25,
                                        val_batches=8,
                                        early_stopping_metric='loss',
                                        patience=300,
                                        min_delta=1e-3,
                                        restore_best=True,
                                        )
        eval_stats = model.evaluate(const_value=1.0, n_batches=5)
        elapsed = time.time() - t0
        log_row = {
            'phase': 'final_finetune',
            'const': 1.0,
            'lr': float(lr),
            'n_iter': int(n_iter),
            'train_last_loss': train_stats['last_loss'],
            'best_iter': train_stats['best_iter'],
            'best_score': train_stats['best_score'],
            'stopped_early': train_stats['stopped_early'],
            'eval_mean_loss': eval_stats['mean_loss'],
            'eval_std_loss': eval_stats['std_loss'],
            'eval_mean_y0': eval_stats['mean_y0'],
            'eval_std_y0': eval_stats['std_y0'],
            'elapsed_sec': float(elapsed),
        }
        stage_logs.append(log_row)
        print(
            f"[FinalSummary] const=1.0, lr={lr:.1e}, iters={n_iter}, "
            f"best_it={train_stats['best_iter']}, best_score={train_stats['best_score']:.3e}, "
            f"eval_loss={eval_stats['mean_loss']:.3e}±{eval_stats['std_loss']:.2e}, "
            f"eval_Y0={eval_stats['mean_y0']:.3f}±{eval_stats['std_y0']:.3f}, "
            f"time={elapsed:.1f}s"
        )
    if log_path:
        with open(log_path + f'/T_{T}_N_{N}_M_{M}_train_{stage_plan * len(coupling_levels) + final_plan}.txt', 'w') as file:
            tot_time = 0
            for row in stage_logs:
                tot_time += row["elapsed_sec"]
                file.write(f"phase={row['phase']}, const={row['const']:.1f}, lr={row['lr']:.1e}, iters={row['n_iter']}, eval_loss={row['eval_mean_loss']:.3e}, eval_y0={row['eval_mean_y0']:.3f}, elapsed_sec={row['elapsed_sec']}\n")
            file.write(f'total time= {tot_time}s = {tot_time/3600}h')
    if plot_path and u_exact and z_exact:
        do_plots(model, plot_path, u_exact, z_exact, stage_plan, coupling_levels, final_plan)
    if save_path:
        model.save_model(f"{save_path}/T_{T}_N_{N}_M_{M}_train_{stage_plan * len(coupling_levels) + final_plan}.ckpt")
    model.sess.close()


def do_plots(model, plot_path, u_exact, z_exact, stage_plan, coupling_levels, final_plan):
    M,N,D,T = model.M,model.N,model.D,model.T
    t_test, W_test, Xi_test = model.fetch_minibatch()
    X_pred, Y_pred, Z_pred = model.predict(Xi_test, t_test, W_test)


    Xi_reshaped = X_pred.reshape(-1, D)
    t_reshaped = t_test.reshape(-1, 1)

    Y_exact = u_exact(t_reshaped, Xi_reshaped).reshape(M, N+1, 1)
    Z_exact = z_exact(t_reshaped, Xi_reshaped).reshape(M, N+1, D)

    # Exact mean of u(0, X0)
    u0_exact = u_exact(
        np.zeros((M,1)),
        Xi_test
    )

    mean_exact_Y0 = np.mean(u0_exact)
    mean_pred_Y0  = np.mean(Y_pred[:,0,0])

    print("\n--- Mean comparison at t=0 ---")
    print(f"Mean Predicted Y0: {mean_pred_Y0:.6f}")
    print(f"Mean Exact Y0:     {mean_exact_Y0:.6f}")
    print(f"Absolute Error:    {abs(np.mean(u0_exact - Y_pred[:,0,0])):.6e}")


    plt.figure(figsize=(10,6))
    plt.plot(t_test[0,:,0], X_pred[0,:,0], 'b', label='S trajectory')
    plt.plot(t_test[0,:,0], X_pred[0,:,1], 'r', label='H trajectory')
    plt.plot(t_test[0,:,0], X_pred[0,:,2], 'y', label='V trajectory')
    plt.plot(t_test[0,:,0], X_pred[0,:,3], 'g', label='X trajectory')
    plt.title("Soluzione Sistema Pienamente Accoppiato 4D")
    plt.xlabel("Tempo t")
    plt.ylabel("A_t")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_path}/T_{T}_N_{N}_M_{M}_train_{stage_plan * len(coupling_levels) + final_plan}_FSDE.png")

    plt.figure(figsize=(10,6))
    plt.plot(t_test[0,:,0], Y_pred[0,:,0], 'b', label='Learned Y')
    plt.plot(t_test[0,:,0], Y_exact[0,:,0], 'r--', label='Exact Y')
    plt.plot(t_test[1:5,:,0].T, Y_pred[1:5,:,0].T, 'b')
    plt.plot(t_test[1:5,:,0].T, Y_exact[1:5,:,0].T, 'r--')
    plt.title("Soluzione Sistema Pienamente Accoppiato 4D")
    plt.xlabel("Tempo t")
    plt.ylabel("Y_t")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_path}/T_{T}_N_{N}_M_{M}_train_{stage_plan * len(coupling_levels) + final_plan}_Y.png")
    
    # Errore relativo medio sui componenti di Z (stile log_z, ma corretto per ogni componente)
    eps = 1e-8
    rel_err_Z = np.abs((Z_pred - Z_exact) / (np.abs(Z_exact) + eps))
    mean_rel_err_Z = np.mean(rel_err_Z, axis=0)  # (N+1, D)
    
    labels = ['Z_S', 'Z_H', 'Z_V', 'Z_X']
    colors = ['b', 'g', 'r', 'm']
    
    plt.figure(figsize=(10,6))
    for d in range(D):
        plt.plot(t_test[0,:,0], mean_rel_err_Z[:, d], colors[d], label=f'Mean Rel Error {labels[d]}')
    plt.yscale('log')
    plt.title('Errore relativo medio di Z (scala log)')
    plt.xlabel('Tempo t')
    plt.ylabel('Errore relativo medio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_path}/T_{T}_N_{N}_M_{M}_train_{stage_plan * len(coupling_levels) + final_plan}_Z.png")
    
    abs_err_Z = np.mean(np.abs(Z_pred - Z_exact),axis=0)
    abs_err_Y = np.mean(np.abs(Y_pred - Y_exact),axis=0)
    plt.figure(figsize=(10,6))
    plt.plot(t_test[0,:,0], abs_err_Y[:,0],'y', label=f'Mean Abs Error Y')
    for d in range(D):
        plt.plot(t_test[0,:,0], abs_err_Z[:, d], colors[d], label=f'Mean Abs Error {labels[d]}')
    plt.yscale('log')
    plt.title('Errore assoluto medio di Y,Z (scala log)')
    plt.xlabel('Tempo t')
    plt.ylabel('Errore assoluto medio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_path}/T_{T}_N_{N}_M_{M}_train_{stage_plan * len(coupling_levels) + final_plan}_abs_err.png")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
