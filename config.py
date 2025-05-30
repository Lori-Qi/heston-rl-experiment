import os
import tensorflow as tf
import keras as k
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
k.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
tf.config.optimizer.set_jit(True)
plt.ioff()