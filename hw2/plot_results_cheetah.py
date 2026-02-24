import os
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_dir(base_dir, prefix):
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    target_start = f"HalfCheetah-v4_{prefix}_sd1_"
    matching_dirs = [d for d in dirs if d.startswith(target_start)]
    if not matching_dirs:
        print(f"No directories found for prefix {prefix}")
        return None
    matching_dirs.sort()
    return os.path.join(base_dir, matching_dirs[-1])

def plot_experiments(exp_names, title, out_path, base_dir="exp", y_key="Eval_AverageReturn"):
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
        
        if "Train_EnvstepsSoFar" in df.columns and y_key in df.columns:
            # For the baseline loss plot, we don't want to plot "cheetah" since it has no baseline loss
            if y_key == "Baseline Loss" and "Baseline Loss" not in df.columns:
                continue
            plt.plot(df["Train_EnvstepsSoFar"], df[y_key], label=exp_name)
        else:
            print(f"Missing required columns in {log_file}")
            
    plt.title(title)
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel(y_key)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    exps = ["cheetah", "cheetah_baseline", "cheetah_baseline_low_lr"]
    plot_experiments(exps, "HalfCheetah Eval Return", "report/cheetah_eval_return.png", y_key="Eval_AverageReturn")
    plot_experiments(["cheetah_baseline", "cheetah_baseline_low_lr"], "HalfCheetah Baseline Loss", "report/cheetah_baseline_loss.png", y_key="Baseline Loss")
