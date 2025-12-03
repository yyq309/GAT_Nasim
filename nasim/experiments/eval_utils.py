import numpy as np


def evaluate_agent(agent, env, num_episodes=50):
    returns, steps, goals = [], [], 0

    for _ in range(num_episodes):
        ep_r, ep_s, goal = agent.run_eval_episode(eval_epsilon=0)
        returns.append(ep_r)
        steps.append(ep_s)
        goals += int(goal)

    return np.mean(returns), np.mean(steps), goals / num_episodes
