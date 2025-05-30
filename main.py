import numpy as np
import pandas as pd 
from training_pipeline import run_heston_rl_experiment 

if __name__ == "__main__":
   
    results_df = run_heston_rl_experiment(
        num_simulation_worlds=100, 
        agent_max_training_eps=1500, 
        wgan_training_iterations=2000, 
        wgan_sim_paths_count=1000 
    )
    print("\nExperiment run finished. Displaying final results DataFrame head:")
    print(results_df.head())
