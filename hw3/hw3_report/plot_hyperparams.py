import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_hyperparams(log_files, labels):
    plt.figure(figsize=(10, 6))

    for log_file, label in zip(log_files, labels):
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            
            if 'Eval_AverageReturn' in df.columns:
                eval_df = df.dropna(subset=['Eval_AverageReturn'])
                plt.plot(eval_df['step'], eval_df['Eval_AverageReturn'], 
                         label=f'Eval Return (TUP={label})', marker='o', linewidth=2, markersize=4)

        else:
            print(f"Error: {log_file} not found.")

    plt.xlabel('Environment Steps')
    plt.ylabel('Return')
    plt.title('Double DQN on LunarLander-v2 (Target Update Period Comparison)')
    plt.axhline(y=200, color='r', linestyle='--', label='Target Return (200)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hw3_report/lunarlander_hyperparams.png')
    print("Plot saved to hw3_report/lunarlander_hyperparams.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_500', type=str, required=True)
    parser.add_argument('--log_1000', type=str, required=True)
    parser.add_argument('--log_2000', type=str, required=True)
    parser.add_argument('--log_5000', type=str, required=True)
    args = parser.parse_args()
    
    logs = [args.log_500, args.log_1000, args.log_2000, args.log_5000]
    labels = ['500', '1000', '2000', '5000']
    
    plot_hyperparams(logs, labels)
