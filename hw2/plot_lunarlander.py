import os
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_dir(base_dir, prefix):
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    target_start = f"LunarLander-v2_{prefix}_sd1_"
    matching_dirs = [d for d in dirs if d.startswith(target_start)]
    if not matching_dirs:
        print(f"No directories found for prefix {prefix}")
        return None
    matching_dirs.sort()
    return os.path.join(base_dir, matching_dirs[-1])

def plot_experiments(exp_names, title, out_path, base_dir="exp"):
    plt.figure(figsize=(10, 6))
    for exp_name in exp_names:
        latest_dir = get_latest_dir(base_dir, exp_name)
        if not latest_dir:
            continue
        
        log_file = os.path.join(latest_dir, "log.csv")
        if not os.path.exists(log_file):
            print(f"Log file missing: {log_file}")
            continue
            
        df = pd.read_csv(log_file)
        
        if "Train_EnvstepsSoFar" in df.columns and "Eval_AverageReturn" in df.columns:
            plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=exp_name)
        else:
            print(f"Missing required columns in {log_file}")
            
    plt.title(title)
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    lunar_lander_exps = [
        "lunar_lander_lambda0",
        "lunar_lander_lambda0.95",
        "lunar_lander_lambda0.98",
        "lunar_lander_lambda0.99",
        "lunar_lander_lambda1"
    ]
    plot_experiments(lunar_lander_exps, "LunarLander GAE Lambda Comparison", "report/lunar_lander.png")
