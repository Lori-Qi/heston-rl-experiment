import tensorflow as tf
import keras as k
import tensorflow_addons as tfa 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.data import Dataset, AUTOTUNE
import time 
from pathlib import Path 
from scipy import stats 
from statsmodels.tsa.stattools import acf 
import json 
import warnings
import config

def g_model(z_dim, d_dim):
    net = k.Sequential()
    net.add(k.Input((z_dim,)))

    initializer = tf.keras.initializers.GlorotNormal()

    net.add(k.layers.Dense(256, 'tanh', kernel_initializer=initializer,
                          kernel_regularizer=k.regularizers.l2(1e-5)))
    net.add(k.layers.Dense(512, 'tanh', kernel_initializer=initializer,
                          kernel_regularizer=k.regularizers.l2(1e-5)))
    net.add(k.layers.Dense(1024, 'tanh', kernel_initializer=initializer))
    net.add(k.layers.Dense(2048, 'tanh', kernel_initializer=initializer))
    net.add(k.layers.Dense(4096, 'tanh', kernel_initializer=initializer))

    net.add(k.layers.Dense(d_dim, activation='tanh', kernel_initializer=initializer))

    net.add(k.layers.Lambda(lambda x: x * 10.0))

    return net

def c_model(d_dim, c_dim):
    net = k.Sequential()
    net.add(k.Input((d_dim,)))

    net.add(k.layers.BatchNormalization())

    net.add(k.layers.Dense(4096, 'tanh'))
    net.add(k.layers.Dropout(0.25))
    net.add(tfa.layers.Maxout(256))

    net.add(k.layers.Dense(64, 'tanh'))
    net.add(k.layers.Dropout(0.25))
    net.add(tfa.layers.Maxout(4))

    net.add(k.layers.Dense(c_dim))
    return net


def simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, dt, num_paths):
    num_steps = int(T / dt)
    lnS = np.full((num_paths, num_steps + 1), np.log(S0), dtype=np.float32)
    v = np.full((num_paths, num_steps + 1), v0, dtype=np.float32)

    Z1 = np.random.randn(num_paths, num_steps)
    Z2 = np.random.randn(num_paths, num_steps)

    sqrt_dt = np.sqrt(dt)

    dW1 = sqrt_dt * Z1
    dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

    for t in range(num_steps):
        v_pos  = np.maximum(v[:, t], 0.0)
        sqrt_v = np.sqrt(v_pos)
        v[:, t+1]   = v[:, t] + kappa*(theta-v_pos)*dt + xi*sqrt_v*dW2[:, t]
        lnS[:, t+1] = lnS[:, t] + (r - 0.5*v_pos)*dt + sqrt_v*dW1[:, t]

    return np.stack([lnS, v], axis=-1)   # shape (N, T+1, 2)


def prepare_data_for_wgan(paths, batch_size):
    # reshape 3D data to 2D
    num_paths, steps_plus_one, features = paths.shape # 2 features
    flattened_paths = paths.reshape(num_paths, -1) # Shape: (num_paths, (num_steps+1)*2)

    print(f"Before normalization - Min: {flattened_paths.min()}, Max: {flattened_paths.max()}")

    mean = np.mean(flattened_paths, axis=0)
    std = np.std(flattened_paths, axis=0)

    min_std = 1e-8
    std = np.maximum(std, min_std)

    normalized_paths = (flattened_paths - mean) / std

    clip_value = 10.0  
    normalized_paths = np.clip(normalized_paths, -clip_value, clip_value)

    print(f"After normalization - Min: {normalized_paths.min()}, Max: {normalized_paths.max()}")

    ds = (
        Dataset.from_tensor_slices(normalized_paths.astype(np.float32))
        .shuffle(buffer_size=num_paths)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    return ds, flattened_paths, mean, std

def inverse_transform(norm_vec, mean, std):
    return norm_vec * std + mean

def gen_z(b_size, z_dim):
    yield tf.random.normal((b_size, z_dim))

def evaluate_distribution(real, generated, title_real="Real", title_gen="Generated"):
    plt.figure(figsize=(10,6))
    sns.kdeplot(real, color='darkred', label=title_real)
    sns.histplot(generated, color='royalblue', stat='density', label=title_gen)
    plt.legend(); plt.title("Distribution Comparison")

def evaluate_moments(real_ret, gen_ret):
    funcs = [("Mean", np.mean), ("Var", np.var),
             ("Skew", stats.skew), ("Kurtosis", stats.kurtosis)]
    for name, fn in funcs:
        print(f"{name:<8} | Real: {fn(real_ret):.4f} | Gen: {fn(gen_ret):.4f}")

def evaluate_acf_mse(real_paths, gen_paths, max_lag=50):
    def avg_acf(paths):
        ac_list = []
        for seq in paths:
            ac_vals = acf(seq, nlags=max_lag, fft=True)[1:]
            if not np.any(np.isnan(ac_vals)):
                ac_list.append(ac_vals)
        return np.mean(ac_list, axis=0) if ac_list else np.zeros(max_lag)

    acf_real = avg_acf(real_paths)
    acf_gen = avg_acf(gen_paths)
    mse = np.mean((acf_real - acf_gen) ** 2)
    print(f"ACF-MSE (lags 1-{max_lag}): {mse:.6f}")

def check_for_explosion(loss_value, prev_losses, threshold=1e6):
    if abs(loss_value) > threshold:
        print(f"WARNING: Loss value {loss_value} exceeds threshold {threshold}!")
        return True

    if len(prev_losses) > 5:
        avg_prev = np.mean(np.abs(prev_losses[-5:]))
        if abs(loss_value) > 10 * avg_prev:
            print(f"WARNING: Loss jumped from average {avg_prev} to {loss_value}!")
            return True

    return False


def monitor_grad_norms(variables):
    total_norm = 0
    for var in variables:
        norm = tf.norm(var)
        total_norm += norm**2
    return tf.sqrt(total_norm)


def grad_penalty(x, y, net):
    eps = tf.random.uniform((tf.shape(x)[0], 1))
    eps = tf.tile(eps, [1, tf.shape(x)[1]])  
    est = eps * x + (1. - eps) * y
    with tf.GradientTape() as m_tape:
        m_tape.watch(est)
        m_grad = m_tape.gradient(net(est), est)
    return tf.reduce_mean(tf.square(tf.norm(m_grad, 2, 1) - 1.))


def c_train(c_net, g_net, c_optm, x, z, gamma):
    y = g_net(z)
    with tf.GradientTape() as c_tape:
        c_x, c_y = c_net(x), c_net(y)
        term_2 = gamma * grad_penalty(x, y, c_net)
        c_loss = -(tf.reduce_mean(c_x) - tf.reduce_mean(c_y)) + term_2
    c_grad = c_tape.gradient(c_loss, c_net.trainable_variables)

    c_grad, _ = tf.clip_by_global_norm(c_grad, 1.0)

    c_optm.apply_gradients(zip(c_grad, c_net.trainable_variables))
    return c_loss


def g_train(c_net, g_net, g_optm, z):
    with tf.GradientTape() as g_tape:
        y = g_net(z)
        c_y = c_net(y)
        g_loss = -tf.reduce_mean(c_y)
    g_grad = g_tape.gradient(g_loss, g_net.trainable_variables)

    g_grad, _ = tf.clip_by_global_norm(g_grad, 1.0)

    g_optm.apply_gradients(zip(g_grad, g_net.trainable_variables))
    return g_loss


@tf.function
def train_itt(c_net, g_net, c_optm, g_optm, c_itts, c_x, c_z, g_z, gamma):
    c_loss = 0.
    for itt in tf.range(c_itts):
        c_loss = c_train(c_net, g_net, c_optm, c_x[itt], c_z[itt], gamma)
    g_loss = g_train(c_net, g_net, g_optm, g_z)
    return c_loss, g_loss


def train_wgan_heston(
        S0=100.0, v0=0.04, r=0.03,
        kappa=3.0, theta=0.05, xi=0.60, rho=-0.60,
        T=1.0, dt=1/252, num_paths=10000,
        z_dim=32, batch_size=64, c_itts=5,
        gamma=10.0, n_itts=2000, initial_lr=1e-5,
        save_dir="./heston_wgan", log_freq=10):

    import numpy as np

    Path(save_dir, "weights").mkdir(parents=True, exist_ok=True)
    real_paths = simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, dt, num_paths)
    num_paths, steps_plus_one, _ = real_paths.shape

    print("Preparing dataset.")
    dataset, flattened_paths, mean, std = prepare_data_for_wgan(real_paths, batch_size)
    d_dim = flattened_paths.shape[1]

    print(f"z_dim = {z_dim}, d_dim = {d_dim}")

    real_data_iter = iter(dataset.repeat())
    print("Creating generator and critic.")
    g_net = g_model(z_dim, d_dim)
    c_net = c_model(d_dim, 1)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=500, decay_rate=0.95, staircase=True)

    g_optm = k.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9)
    c_optm = k.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9)

    _ = g_net(tf.zeros((1, z_dim), dtype=tf.float32), training=False)
    _ = c_net(tf.zeros((1, d_dim), dtype=tf.float32), training=False)

    zero_g = [tf.zeros_like(w) for w in g_net.trainable_variables]
    zero_c = [tf.zeros_like(w) for w in c_net.trainable_variables]

    g_optm.apply_gradients(zip(zero_g, g_net.trainable_variables))
    c_optm.apply_gradients(zip(zero_c, c_net.trainable_variables))

    print("Training started.")
    c_losses, g_losses = [], []
    start_time = time.time()

    for itt in range(1, n_itts + 1):
        real_batch = tf.stack([next(real_data_iter) for _ in range(c_itts)])
        c_z_batch = tf.stack([tf.random.normal((batch_size, z_dim)) for _ in range(c_itts)])
        g_z_batch = tf.random.normal((batch_size, z_dim))

        c_loss, g_loss = train_itt(c_net, g_net, c_optm, g_optm, c_itts,
                                   real_batch, c_z_batch, g_z_batch, gamma)

        if tf.math.is_nan(c_loss) or tf.math.is_nan(g_loss):
            print(f"[STOP] NaN loss at iter {itt}")
            break

        if check_for_explosion(float(c_loss), c_losses) or check_for_explosion(float(g_loss), g_losses):
            print(f"[WARN] Loss exploded at iter {itt}, reducing LR")
            current_g_lr = g_optm.learning_rate
            current_c_lr = c_optm.learning_rate

            if isinstance(current_g_lr, tf.keras.optimizers.schedules.ExponentialDecay):
                new_g_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    current_g_lr(tf.cast(g_optm.iterations, tf.int64)) * 0.5,
                    decay_steps=500, decay_rate=0.95, staircase=True)
                new_c_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    current_c_lr(tf.cast(c_optm.iterations, tf.int64)) * 0.5,
                    decay_steps=500, decay_rate=0.95, staircase=True)
                g_optm.learning_rate = new_g_lr
                c_optm.learning_rate = new_c_lr
            else:
                g_optm.learning_rate.assign(current_g_lr * 0.5)
                c_optm.learning_rate.assign(current_c_lr * 0.5)

        c_losses.append(float(c_loss))
        g_losses.append(float(g_loss))

        if itt % 100 == 0:
            g_net.save_weights(Path(save_dir, "weights", f"g_{itt:04d}.h5"))

        if itt % log_freq == 0:
            if isinstance(g_optm.learning_rate, tf.keras.optimizers.schedules.ExponentialDecay):
                current_lr = g_optm.learning_rate(tf.cast(g_optm.iterations, tf.int64))
            else:
                current_lr = g_optm.learning_rate

            print(f"Iter {itt:4d}: C_loss={c_loss:.4f}, G_loss={g_loss:.4f}, LR={float(current_lr):.6f}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(c_losses)
    plt.title('Critic Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(g_losses)
    plt.title('Generator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

    import json
    g_net.save(Path(save_dir, "generator.h5"))
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(Path(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f)

    print(f"Generator and stats saved to {save_dir}")

    history = {"c_loss": c_losses, "g_loss": g_losses}
    return g_net, history, real_paths, mean, std