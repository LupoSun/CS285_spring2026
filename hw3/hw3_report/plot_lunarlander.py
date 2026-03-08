import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_lunarlander(log_file):
    plt.figure(figsize=(10, 6))

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        
        if 'Eval_AverageReturn' in df.columns:
            eval_df = df.dropna(subset=['Eval_AverageReturn'])
            plt.plot(eval_df['step'], eval_df['Eval_AverageReturn'], 
                     label='Eval Return', color='blue', marker='o', linewidth=2)
                     
    else:
        print(f"Error: {log_file} not found.")
        return

    plt.xlabel('Environment Steps')
    plt.ylabel('Return')
    plt.title('Double DQN on LunarLander-v2')
    plt.axhline(y=200, color='r', linestyle='--', label='Target Return (200)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hw3_report/lunarlander_ddqn.png')
    print("Plot saved to hw3_report/lunarlander_ddqn.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    args = parser.parse_args()
    
    plot_lunarlander(args.log)
