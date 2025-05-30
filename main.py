import numpy as np
import pandas as pd 
from training_pipeline import run_heston_rl_experiment 

if __name__ == "__main__":
   
    results_df = run_heston_rl_experiment(
       num_worlds=100,
       rl_max_train_episodes=1500,
       wgan_training_itts=2000,
       wgan_num_paths=1000
)
    print("\nExperiment run finished. Displaying final results DataFrame head:")
    print(results_df.head())
