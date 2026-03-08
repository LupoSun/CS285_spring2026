import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_mspacman(log_file):
    plt.figure(figsize=(10, 6))

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        
        if 'Eval_AverageReturn' in df.columns:
            eval_df = df.dropna(subset=['Eval_AverageReturn'])
            plt.plot(eval_df['step'], eval_df['Eval_AverageReturn'], 
                     label='MsPacman Eval Return', color='blue', marker='o', linewidth=2)
                     
        if 'Train_EpisodeReturn' in df.columns:
            train_df = df.dropna(subset=['Train_EpisodeReturn'])
            plt.plot(train_df['step'], train_df['Train_EpisodeReturn'].rolling(100).mean(), 
                     label='MsPacman Train Return (Moving Avg 100)', color='orange', alpha=0.5)
    else:
        print(f"Error: {log_file} not found.")
        return

    plt.xlabel('Environment Steps')
    plt.ylabel('Return')
    plt.title('DQN on MsPacman-v0 (Train vs Eval Return)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hw3_report/mspacman_return.png')
    print("Plot saved to hw3_report/mspacman_return.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    args = parser.parse_args()
    
    plot_mspacman(args.log)
