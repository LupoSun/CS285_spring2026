# Homework 3: Deep Reinforcement Learning

## Section 2.4 - DQN Agent on CartPole

### Deliverables

Below is the plot showing both the training and evaluation returns for the DQN agent running on the `CartPole-v1` environment for 100,000 steps.

As required, I plot the `Eval Average Return` (blue line) and the rolling 100-step average `Train Episode Return` (orange line).

![Cartpole Learning Curve](cartpole_learning_curve.png)

### Summary

The agent successfully climbs towards the maximum reward of 500, demonstrating that the epsilon-greedy policy, the target critic updates, and the Q-learning Bellman backups have been implemented correctly.
