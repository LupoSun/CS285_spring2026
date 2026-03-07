import pandas as pd
import matplotlib.pyplot as plt
import os

log_file = 'exp/CartPole-v1_dqn_sd1_20260306_192519/log.csv'
df = pd.read_csv(log_file)

plt.figure(figsize=(10, 6))

if 'Eval_AverageReturn' in df.columns:
    eval_df = df.dropna(subset=['Eval_AverageReturn'])
    plt.plot(eval_df['step'], eval_df['Eval_AverageReturn'], label='Eval Average Return', marker='o', linewidth=2)

if 'Train_EpisodeReturn' in df.columns:
    train_df = df.dropna(subset=['Train_EpisodeReturn'])
    plt.plot(train_df['step'], train_df['Train_EpisodeReturn'].rolling(100).mean(), label='Train Episode Return (Moving Avg 100)', alpha=0.5)

plt.xlabel('Environment Steps')
plt.ylabel('Return')
plt.title('DQN on CartPole-v1 Learning Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cartpole_learning_curve.png')
print("Plot saved to cartpole_learning_curve.png")
