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
    singleq_dir = find_latest_log_dir("Hopper-v4_sac_singleq_sd1_*")
    clipq_dir = find_latest_log_dir("Hopper-v4_sac_clipq_sd1_*")

    if not singleq_dir or not clipq_dir:
        print("Could not find both singleq and clipq runs in your exp folder.")
        return

    print(f"Loading single-Q logs from: {singleq_dir}")
    print(f"Loading clipped double-Q logs from: {clipq_dir}")

    singleq_csv = os.path.join(singleq_dir, "log.csv")
    clipq_csv = os.path.join(clipq_dir, "log.csv")

    singleq_df = pd.read_csv(singleq_csv)
    clipq_df = pd.read_csv(clipq_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Eval returns
    if 'step' in singleq_df.columns and 'Eval_AverageReturn' in singleq_df.columns:
        valid_sq = singleq_df.dropna(subset=['step', 'Eval_AverageReturn'])
        ax1.plot(valid_sq["step"], valid_sq["Eval_AverageReturn"], label="Single-Q", color="steelblue")
        
    if 'step' in clipq_df.columns and 'Eval_AverageReturn' in clipq_df.columns:
        valid_cq = clipq_df.dropna(subset=['step', 'Eval_AverageReturn'])
        ax1.plot(valid_cq["step"], valid_cq["Eval_AverageReturn"], label="Clipped Double-Q (Min)", color="darkorange")
    
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Eval Average Return")
    ax1.set_title("Eval Return on Hopper-v4")
    ax1.axhline(1500, color='red', linestyle='--', alpha=0.5, label='Target Return (1500)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q-values
    if 'step' in singleq_df.columns and 'q_values' in singleq_df.columns:
        valid_sq_q = singleq_df.dropna(subset=['step', 'q_values'])
        ax2.plot(valid_sq_q["step"], valid_sq_q["q_values"], label="Single-Q", color="steelblue")
        
    if 'step' in clipq_df.columns and 'q_values' in clipq_df.columns:
        valid_cq_q = clipq_df.dropna(subset=['step', 'q_values'])
        ax2.plot(valid_cq_q["step"], valid_cq_q["q_values"], label="Clipped Double-Q (Min)", color="darkorange")
        
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Average Q-values")
    ax2.set_title("Q-values over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    os.makedirs("hw3_report", exist_ok=True)
    out_path = "hw3_report/hopper_sac_qbackup.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
