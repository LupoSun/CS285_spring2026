import os
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_dir(base_dir, prefix):
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    # Match EXACTLY the prefix up to _sd1
    # Example: CartPole-v0_cartpole_sd1_...
    # prefix is like "cartpole"
    target_start = f"CartPole-v0_{prefix}_sd1_"
    matching_dirs = [d for d in dirs if d.startswith(target_start)]
    if not matching_dirs:
        print(f"No directories found for prefix {prefix}")
        return None
    # sorts by datetime suffix
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
        
        # We need Train_EnvstepsSoFar on X and Eval_AverageReturn on Y
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
    small_batch_exps = ["cartpole", "cartpole_rtg", "cartpole_na", "cartpole_rtg_na"]
    plot_experiments(small_batch_exps, "CartPole Small Batch (1000) Learning Curves", "report/cartpole_small_batch.png")
    
    large_batch_exps = ["cartpole_lb", "cartpole_lb_rtg", "cartpole_lb_na", "cartpole_lb_rtg_na"]
    plot_experiments(large_batch_exps, "CartPole Large Batch (4000) Learning Curves", "report/cartpole_large_batch.png")
