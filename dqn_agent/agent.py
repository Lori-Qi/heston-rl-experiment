import tensorflow as tf
import keras as k
import numpy as np
import os 
import warnings 
from .networks import create_q_network 
from .replay_buffer import TFReplayBuffer 

class DQNAgent:
    def __init__(
        self,
        state_size,
        action_space=np.linspace(-1, 1, 41),
        memory_size=2000,
        gamma=0.97,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        batch_size=64,
        target_update_freq=100,
    ):
        self.state_size = state_size
        self.action_space = np.array(action_space, dtype=np.float32)
        self.action_size = len(self.action_space)

        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        self.qnetwork_local = create_q_network(self.state_size, self.action_size)
        self.qnetwork_target = create_q_network(self.state_size, self.action_size)
        self._update_target_network()

        self.optimizer = k.optimizers.Adam(learning_rate=self.learning_rate)
        self.memory = TFReplayBuffer(memory_size, self.batch_size)

        self.t_step = 0
        self.loss_history = []
        self.epsilon_history = []

    def select_action(self, state):
        if tf.random.uniform(()) <= self.epsilon:
            action_idx = tf.random.uniform((), 0, self.action_size, dtype=tf.int64)
        else:
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            q_values = self.qnetwork_local(state_tensor, training=False)
            action_idx = tf.argmax(q_values[0]).numpy()
        return self.action_space[action_idx], int(action_idx)

    def step(self, state, action_idx, reward, next_state, done):
        self.memory.add(state, action_idx, reward, next_state, done)
        self.t_step += 1

        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            if experiences:
                loss = self._learn(experiences)
                if loss is not None:
                    self.loss_history.append(loss.numpy())

        if self.t_step % self.target_update_freq == 0:
            self._update_target_network()

        self._decay_epsilon()
        self.epsilon_history.append(self.epsilon)

    def _learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_next_all = self.qnetwork_target(next_states, training=False)
        q_next_max = tf.reduce_max(q_next_all, axis=1, keepdims=True)
        q_targets = rewards + (self.gamma * q_next_max * (1.0 - dones))

        with tf.GradientTape() as tape:
            q_local_all = self.qnetwork_local(states, training=True)
            gather_idx = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int64), tf.squeeze(actions)], axis=1)
            q_expected = tf.expand_dims(tf.gather_nd(q_local_all, gather_idx), axis=1)
            loss = tf.keras.losses.MeanSquaredError()(y_true=q_targets, y_pred=q_expected)

        grads = tape.gradient(loss, self.qnetwork_local.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.qnetwork_local.trainable_variables))
        return loss

    def _update_target_network(self):
        self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_weights(self, filepath_prefix):
        try:
            self.qnetwork_local.save_weights(filepath_prefix)
            print(f"Agent weights saved to {filepath_prefix}.*")
        except Exception as e:
            print(f"Error saving agent weights: {e}")

    def load_weights(self, filepath_prefix):
        p_idx = filepath_prefix + ".index"
        p_h5 = filepath_prefix + ".weights.h5"
        p_keras = filepath_prefix + ".keras"
        path = None
        if os.path.exists(p_idx):
            path = filepath_prefix
        elif os.path.exists(p_h5):
            path = p_h5
        elif os.path.exists(p_keras):
            path = p_keras
        else:
            print(f"Weights not found: {filepath_prefix}")
            return False
        try:
            self.qnetwork_local.load_weights(path)
            self._update_target_network()
            print(f"Agent weights loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading weights from {path}: {e}")
            return False

    def get_metric(self):
        return {
            "loss_history": self.loss_history.copy(),
            "epsilon_history": self.epsilon_history.copy(),
        }