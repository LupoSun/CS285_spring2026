import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def find_latest_log_dir(base_name):
    # Find directories matching the base pattern
    dirs = glob.glob(f"exp/{base_name}")
    if not dirs:
        return None
    # Sort by creation/modification time to get the latest
    return sorted(dirs, key=os.path.getmtime)[-1]

def main():
    fixed_dir = find_latest_log_dir("HalfCheetah-v4_sac_sd1_*")
    auto_dir = find_latest_log_dir("HalfCheetah-v4_sac_autotune_sd1_*")

    if not fixed_dir or not auto_dir:
        print("Could not find both fixed and auto runs in your exp folder.")
        if not fixed_dir:
            print("Missing: Fixed temperature run (HalfCheetah-v4_sac_sd1_*)")
        if not auto_dir:
            print("Missing: Auto-tuned temperature run (HalfCheetah-v4_sac_autotune_sd1_*)")
        return

    print(f"Loading fixed temperature logs from: {fixed_dir}")
    print(f"Loading auto-tuned temperature logs from: {auto_dir}")

    fixed_csv = os.path.join(fixed_dir, "log.csv")
    auto_csv = os.path.join(auto_dir, "log.csv")

    fixed_df = pd.read_csv(fixed_csv)
    auto_df = pd.read_csv(auto_csv)

    # Need 'step' and 'Eval_AverageReturn' for both, 'temperature' for auto
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Eval returns
    if 'step' in fixed_df.columns and 'Eval_AverageReturn' in fixed_df.columns:
        valid_fixed = fixed_df.dropna(subset=['step', 'Eval_AverageReturn'])
        ax1.plot(valid_fixed["step"], valid_fixed["Eval_AverageReturn"], label="Fixed Temperature", color="steelblue")
    if 'step' in auto_df.columns and 'Eval_AverageReturn' in auto_df.columns:
        valid_auto = auto_df.dropna(subset=['step', 'Eval_AverageReturn'])
        ax1.plot(valid_auto["step"], valid_auto["Eval_AverageReturn"], label="Auto-Tuned", color="darkorange")
    
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Eval Average Return")
    ax1.set_title("Eval Return on HalfCheetah-v4")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Temperature
    if 'step' in auto_df.columns and 'temperature' in auto_df.columns:
        valid_temp = auto_df.dropna(subset=['step', 'temperature'])
        ax2.plot(valid_temp["step"], valid_temp["temperature"], label="Learned Alpha (\u03B1)", color="firebrick")
        
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Temperature (\u03B1)")
    ax2.set_title("Auto-Tuned Temperature over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    os.makedirs("hw3_report", exist_ok=True)
    out_path = "hw3_report/halfcheetah_sac_autotune.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
