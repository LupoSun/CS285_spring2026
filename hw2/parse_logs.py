import os
import matplotlib.pyplot as plt

def parse_log(filename):
    steps = []
    returns = []
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_return = None
    for line in lines:
        if line.startswith("Eval_AverageReturn"):
            current_return = float(line.split(":")[1].strip())
        elif line.startswith("Train_EnvstepsSoFar"):
            current_steps = int(line.split(":")[1].strip())
            if current_return is not None:
                steps.append(current_steps)
                returns.append(current_return)
                current_return = None
    return steps, returns

def_steps, def_returns = parse_log("pendulum_default.log")
tune4_steps, tune4_returns = parse_log("pendulum_tune4.log")
tune6_steps, tune6_returns = parse_log("pendulum_tune6.log")

print(f"Default max return: {max(def_returns)} at step {def_steps[def_returns.index(max(def_returns))]}")

def first_1000(steps, returns):
    for s, r in zip(steps, returns):
        if r >= 1000.0:
            return s
    return None

print(f"Tune4 first 1000 at step: {first_1000(tune4_steps, tune4_returns)}")
print(f"Tune6 first 1000 at step: {first_1000(tune6_steps, tune6_returns)}")

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(def_steps, def_returns, label='Default (-n 100, -b 5000, -eb 1000)')
plt.plot(tune4_steps, tune4_returns, label='Tune4 (1K batch, RTG, NA, Baseline)')
plt.plot(tune6_steps, tune6_returns, label='Tune6 (500 batch, RTG, NA, Baseline)')
plt.xlabel('Environment Steps')
plt.ylabel('Average Evaluation Return')
plt.title('InvertedPendulum-v4 Learning Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('report/pendulum_tuning.png')
print("Saved plot to report/pendulum_tuning.png")
