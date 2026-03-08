# Homework 3: Deep Reinforcement Learning

## Section 2.4 - DQN Agent on CartPole

### Deliverables

Below is the plot showing both the training and evaluation returns for the DQN agent running on the `CartPole-v1` environment for 100,000 steps.

As required, I plot the `Eval Average Return` (blue line) and the rolling 100-step average `Train Episode Return` (orange line).

![Cartpole Learning Curve](cartpole_learning_curve.png)

### Summary

The agent successfully climbs towards the maximum reward of 500, demonstrating that the epsilon-greedy policy, the target critic updates, and the Q-learning Bellman backups have been implemented correctly.

## Section 2.5 - Double Q-Learning

### LunarLander-v2

Below is the plot for Double DQN on the `LunarLander-v2` environment over 500,000 steps. 

As required, I plot the `Eval Return` (blue line) over the training steps. Notice that the agent reaches a target return of 200 during training.

![LunarLander Double DQN](lunarlander_ddqn.png)

### MsPacman-v0

The default configuration for `MsPacman-v0` utilizes Double Q-Learning. Below is the plot depicting both the evaluation and average training returns over 1,000,000 steps. 

![MsPacman Returns](mspacman_return.png)

**Explanation of difference between Training and Eval Returns early in training:**
Early in training, the *training return* is typically much higher or substantially different from the *eval return*. This occurs because training employs an epsilon-greedy policy with a high $\epsilon$ value (encouraging exploration). MsPacman is an environment where taking random actions quickly leads to death strings and lower scores, whereas the evaluation policy is fully greedy ($\epsilon = 0$), allowing the agent to systematically exploit its current (albeit limited) knowledge without making forced random fatal mistakes. Later in training, as $\epsilon$ decays towards 0, the training and evaluation policies become nearly identical, causing their returns to converge.

## Section 2.6 - Experimenting with Hyperparameters

For this section, I explored the sensitivity of the Double Q-Learning agent on `LunarLander-v2` to the **Target Network Update Frequency** (`target_update_period`). 

I chose this hyperparameter because taking the argmax over un-updated target Q-values is the specific feature implemented by Double DQN to prevent overestimation divergence. Intuitively, updating the target network too rarely ($> 5000$) might result in stable yet agonizingly slow learning progress because the target is essentially frozen. Conversely, updating the network too frequently ($\le 500$) risks removing the stationary stability buffer necessary for Q-values to properly converge without chasing a rapidly fluctuating target. 

Below is the graph plotting four variations of `target_update_period` (500, 1000, 2000, 5000) overlaid on top of each other.

![Hyperparameter Target Period Search](lunarlander_hyperparams.png)

From the graph, we can see that:
- **TUP = 1000 and 2000** provided the most stable trajectory towards the 200 point objective. 
- **TUP = 500** initially learned quickly but often exhibited high variance and instability due to rapidly shifting target Q-values.
- **TUP = 5000** showed smoother behavior, but plateaued early and learned much slower overall because the learning targets remained frozen for too long between updates.
