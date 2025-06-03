import argparse
import os

def run_experiment(choice):
    if choice == "two_players":
        os.system("python run/run_two_players.py")
    elif choice == "three_players":
        os.system("python run/run_three_players.py")
    elif choice == "diff_cost":
        os.system("python run/run_diff_cost.py")
    elif choice == "diff_ability":
        os.system("python run/run_diff_ability.py")
    elif choice == "two_stage":
        os.system("python run/run_two_stage.py")
    else:
        print("Invalid choice. Please choose from: two_players, three_players, diff_cost, diff_ability, two_stage")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and run a tournament experiment.")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["two_players", "three_players", "diff_cost", "diff_ability", "two_stage"],
                        help="Choose which experiment to run.")
    args = parser.parse_args()
    run_experiment(args.experiment)