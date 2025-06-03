# Tournament Experiment
This repository implements a set of game-theoretic simulation experiments, focusing on players' effort allocation and utility computation under various settings. It supports both one-stage and two-stage environments, multiple players, heterogeneous parameters, and different optimization strategies including Gradient Descent, REINFORCE, and PPO.
## Project Structure
tournament_experiment/
├── config/              # Experiment configurations
├── envs/                # Game environments (one-stage, two-stage)
├── agents/              # Solvers and policy optimization agents
├── run/                 # Experiment entry points
├── results/             # Output tables and logs
├── utils/               # Logging, evaluation, and plotting tools
└── main.py              # Master controller for experiment selection
## Installation
We recommend using Python 3.8+ with a virtual environment:
```bash
git clone https://github.com/your_username/tournament_experiment.git
cd tournament_experiment

# Install dependencies
pip install -r requirements.txt

How to Run

Each experiment can be executed directly via the scripts in the run/ directory:

1. One-Stage Game with Two Identical Players

python run/run_two_players.py

2. One-Stage Game with Three Identical Players

python run/run_three_players.py

3. Different Cost Parameters ($k_1 \ne k_2$)

python run/run_diff_cost.py

4. Different Ability Parameters ($l_1 > l_2$)

python run/run_diff_ability.py

5. Two-Stage Game

python run/run_two_stage.py

Alternatively, you can use the master script to choose an experiment interactively:

python main.py

Supported Solvers
	•	Gradient-based Optimizer: agents/gradient_solver.py
	•	REINFORCE: agents/reinforce_agent.py
	•	PPO: agents/ppo_agent.py

You can modify hyperparameters such as learning rate, exploration noise, and update intervals in the corresponding config files.

Output Format
	•	Numerical results are written to results/tables/*.csv
	•	Training logs are saved under results/logs/
	•	Visualization utilities are available in utils/plot.py

Customization

To create a new experiment:
	1.	Create a new configuration file under config/
	2.	Add a corresponding run script in run/

Contact

For questions or contributions, please reach out to: fjiang4@student.gsu.edu