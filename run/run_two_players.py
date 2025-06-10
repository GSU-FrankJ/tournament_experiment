import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.one_stage_two_players import config
from envs.one_stage_env import OneStageEnv
from agents.gradient_solver import gradient_descent_solver
from utils.logger import save_result
from agents.reinforce_agent import REINFORCEAgent

env = OneStageEnv(config)
effort, eu, cost = gradient_descent_solver(env)  # No need to capture return values

result = {
    "k": config["k"],
    "q": config["q"],
    "w_h": config["w_h"],
    "w_l": config["w_l"],
    "EU": round(config["eu"], 2),
    "Cost of effort": round(config["cost"], 2),
    "effort": round(config["effort"], 2),
    "Model training": "Gradient",
    "Parameter": "",
    "Effort [0,100]": round(effort, 2)if config["effort_range"][1] == 100 else "",
    "Effort [0,200]": round(effort, 2)if config["effort_range"][1] == 200 else "",
}
save_result(result, "results/tables/two_players.csv")

# === REINFORCE Training ===
import torch
env = OneStageEnv(config)
agent1 = REINFORCEAgent(effort_range=config["effort_range"], log_path="results/logs/reinforce_agent1.csv")
agent2 = REINFORCEAgent(effort_range=config["effort_range"], log_path="results/logs/reinforce_agent2.csv")
num_episodes = 200000

for episode in range(num_episodes):
    state1, state2 = env.reset()
    a1 = agent1.select_action(state1)
    a2 = agent2.select_action(state2)
    _, rewards, _, _, info = env.step(torch.stack([a1, a2]))
    agent1.store_reward(rewards[0])
    agent2.store_reward(rewards[1])
    agent1.update_policy(episode=episode, last_effort=a1)
    agent2.update_policy(episode=episode, last_effort=a2)

final_effort1 = info["efforts"][0]
result_reinforce = {
    "k": config["k"],
    "q": config["q"],
    "w_h": config["w_h"],
    "w_l": config["w_l"],
    "EU": round(config["eu"], 2),
    "Cost of effort": round(config["cost"], 2),
    "effort": round(config["effort"], 2),
    "Model training": "REINFORCE",
    "Parameter": f"episodes={num_episodes}",
    "Effort [0,100]": round(final_effort1, 2) if config["effort_range"][1] == 100 else "",
    "Effort [0,200]": round(final_effort1, 2) if config["effort_range"][1] == 200 else "",
}
save_result(result_reinforce, "results/tables/two_players.csv")