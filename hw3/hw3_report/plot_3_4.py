import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def find_latest_log_dir(base_name):
    dirs = glob.glob(f"exp/{base_name}")
    if not dirs:
        return None
    return sorted(dirs, key=os.path.getmtime)[-1]

def main():
    fixed_dir = find_latest_log_dir("HalfCheetah-v4_sac_sd1_*")

    if not fixed_dir:
        print("Could not find the fixed temperature run (exp_name=sac) for HalfCheetah-v4.")
        return

    print(f"Loading fixed temperature logs from: {fixed_dir}")

    fixed_csv = os.path.join(fixed_dir, "log.csv")
    fixed_df = pd.read_csv(fixed_csv)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    if 'step' in fixed_df.columns and 'Eval_AverageReturn' in fixed_df.columns:
        valid_fixed = fixed_df.dropna(subset=['step', 'Eval_AverageReturn'])
        ax1.plot(valid_fixed["step"], valid_fixed["Eval_AverageReturn"], label="SAC (fixed \u03B1=0.1)", color="steelblue")
    
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Eval Average Return")
    ax1.set_title("Soft Actor-Critic on HalfCheetah-v4")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    
    os.makedirs("hw3_report", exist_ok=True)
    out_path = "hw3_report/halfcheetah_baseline.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
