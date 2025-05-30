import tensorflow as tf 
import numpy as np
from collections import deque, namedtuple 
import random 

class TFReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action_idx", "reward", "next_state", "done"])

    def add(self, state, action_idx, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action_idx = int(action_idx)
        reward = float(reward)
        done = bool(done)

        e = self.experience(state, action_idx, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None

        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.convert_to_tensor([e.state for e in experiences], dtype=tf.float32)
        actions = tf.convert_to_tensor([e.action_idx for e in experiences], dtype=tf.int64)
        rewards = tf.convert_to_tensor([e.reward for e in experiences], dtype=tf.float32)
        next_states = tf.convert_to_tensor([e.next_state for e in experiences], dtype=tf.float32)

        dones = tf.convert_to_tensor([e.done for e in experiences], dtype=tf.float32)

        # Reshape
        actions = tf.reshape(actions, (self.batch_size, 1))
        rewards = tf.reshape(rewards, (self.batch_size, 1))
        dones = tf.reshape(dones, (self.batch_size, 1))

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)