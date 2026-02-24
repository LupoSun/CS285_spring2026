import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Find all run logs
log_dirs = glob.glob("exp/LunarLander-v2_lunar_lander_lambda*")

plt.figure(figsize=(10, 6))

for log_dir in log_dirs:
    log_file = os.path.join(log_dir, "log.csv")
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        
        # Extract lambda value from dirname for the label
        # Example: exp/LunarLander-v2_lunar_lander_lambda0.99_sd1_...
        dirname = os.path.basename(log_dir)
        lambda_str = dirname.split("lambda")[1].split("_sd")[0]
        
        plt.plot(df['Train_EnvstepsSoFar'], df['Eval_AverageReturn'], label=f"$\lambda={lambda_str}$")

plt.title("LunarLander-v2 Evaluation Return with GAE")
plt.xlabel("Environment Steps")
plt.ylabel("Evaluation Average Return")
plt.legend()
plt.grid(True)
plt.savefig("report/lunarlander_gae.png")
print("Plot saved to report/lunarlander_gae.png")
