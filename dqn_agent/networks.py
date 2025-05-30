import tensorflow as tf
import keras as k
from tensorflow.keras import layers 

def create_q_network(state_dim, num_actions, fc1_units=24, fc2_units=24):
    q_network_model = k.Sequential([
        k.Input(shape=(state_dim,), name='state_input'), 
        layers.Dense(fc1_units, activation='relu', name='fc1'),
        layers.Dense(fc2_units, activation='relu', name='fc2'),
        layers.Dense(num_actions, activation=None, name='q_output') 
    ], name='Q_Network')
    return q_network_model