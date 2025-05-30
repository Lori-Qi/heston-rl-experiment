## File Structure

The project is organized into several Python files and subdirectories as shown below.
This structure reflects the placement of the WGAN models and training logic directly in `models_and_trainer.py` at the project root.

```plaintext
project_root/
├── main.py                     # Main entry point to run experiments
├── config.py                   # Global configurations, constants
├── environment.py              # Defines the Heston_Env class (the trading environment)
├── simulators.py               # Contains all price simulator classes
├── evaluation_metrics.py       # Functions for calculating performance metrics
├── models_and_trainer.py       # WGAN generator/critic model definitions and training logic
│
├── dqn_agent/                  # Package for DQN Agent components
│   ├── __init__.py             # Marks dqn_agent as a Python package
│   ├── networks.py               # Q-network architecture
│   ├── replay_buffer.py          # Replay buffer implementation
│   └── agent.py                  # DQNAgent class definition
│
└── training_pipeline.py        # Contains TrainingManager class and run_heston_rl_experiment function
