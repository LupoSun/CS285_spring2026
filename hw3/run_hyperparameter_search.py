import os
import subprocess

TARGET_UPDATES = [500, 1000, 2000, 5000]
BASE_CONFIG_PATH = "experiments/dqn/lunarlander_ddqn.yaml"
OUTPUT_DIR = "experiments/dqn/hyperparams"

def run_hyperparams():
    # We will use the lunarlander double Q config as the base
    with open(BASE_CONFIG_PATH, "r") as f:
        base_content = f.read()

    for tup in TARGET_UPDATES:
        config_path = os.path.join(OUTPUT_DIR, f"lunarlander_ddqn_tup{tup}.yaml")
        
        # Replace the target update period in the config content
        lines = base_content.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith("target_update_period:"):
                new_lines.append(f"target_update_period: {tup}")
            elif line.startswith("exp_name:"):
                new_lines.append(f"exp_name: dqn_tup{tup}")
            else:
                new_lines.append(line)
                
        with open(config_path, "w") as f:
            f.write('\n'.join(new_lines))
            
        print(f"Generated config: {config_path}")
        
        # Launch to modal removed.
        # We will launch manually.

if __name__ == "__main__":
    run_hyperparams()
