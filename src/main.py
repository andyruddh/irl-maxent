#!/usr/bin/env python

import gridworld as W
import maxent as M
import plot as P
import trajectory as T
import solver as S
import optimizer as O
import mce_irl as I
# import frozenlake_mce_irl as I

import numpy as np
import matplotlib.pyplot as plt
import os

def get_policy(start):
    pass

def sttl_goal_reach(D, goal):
    '''
        F[0, T] (Pr[pos = goal] > threshold)
    '''
    goal = goal[0]
    end_time = 12
    threshold = 0.99

    formula = "F[0,{}] (Pr[pos = goal] > {})".format(end_time, threshold)
    print("StTL goal-reach formula: ", formula)

    count = 0
    n_samples = len(D)
    for sample in D:
        visited = []
        transitions = sample._t
        n = len(transitions)
        for i in range(n):
            visited.append(transitions[i][0])
        visited.append(transitions[-1][-1])
        # print(transitions)
        # print(visited)
        idx = visited.index(goal)
        if idx <= end_time:
            count += 1
    
    prob = (count / n_samples)
    rho = prob - threshold

    # print("Probability of reaching the goal: %.2f" % (prob))

    return rho

def get_d2obs(policy, avoid_states):
    d2obs = []
    for s in policy:
        dmin = float("inf")
        for obs in avoid_states:
            d = abs(s - obs)
            dmin = min(dmin, d)
        d2obs.append(dmin)
    return d2obs

def sttl_obstacle_avoid(D, goal, avoid_states):
    '''
        G (Pr[dist2obs > dmin] > 0.99)
    '''
    goal = goal[0]
    threshold = 0.99
    dmin = 2

    formula = "G (Pr[dist2obs > {}] > {})".format(dmin, threshold)
    print("StTL obstacle-avoidance formula: ", formula)

    count = 0
    n_samples = len(D)
    for sample in D:
        visited = []
        transitions = sample._t
        n = len(transitions)
        for i in range(n):
            visited.append(transitions[i][0])
        visited.append(transitions[-1][-1])
        # print(transitions)
        # print(visited)
        d2obs = np.array(get_d2obs(visited, avoid_states))
        d = np.min(d2obs - dmin)
        if d <= 0:
            count += 1

    prob = (count / n_samples)
    rho = prob - threshold

    # print("Probability of reaching the goal: %.2f" % (prob))

    return rho

def main():
    # Grid-world setup parameters
    grid_size = 7 # grid-world size
    p_slip = 0.000003 # slip. with probability p_slip, agent chooses other 3 actions. Default 0.3
    avoid_states = [0, 7, 9, 11] # for 4x4 Frozenlake
    avoid_states = [2, 3, 4, 9, 10, 11, 37, 38, 39, 44, 45, 46]    
    '''
    Ground-truth MDP reward:
        1. reaching the goal gets a reward of +10
        2. reaching avoid-region gets reward 0
        3. any other state gets reward of +1
    NOTE: This somehow does not learn with negative rewards (idk why),
    so I manually designed the ground-truth rewards.
    '''
    # Generate MCE-IRL rewards from demonstrations
    reward_mce, world, goal = I.mce_irl(grid_size, p_slip, avoid_states)

    discount = 0.99 # same as gamma for value iteration. Default 0.7
    value = S.value_iteration(world.p_transition, reward_mce, discount)
    weighting = lambda x: x**1 # giving importance to sub-optimal actions
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)

    # print("Value\n", value)
    # print("Policy\n", policy)
    # print("Policy Exec\n", policy_exec)

    # Randomly sample policies from the learned reward function
    # D = []
    # np.random.seed(0)
    # n_samples = 20
    # for _ in range(n_samples):
    #     start = np.random.randint(0, 16)
    #     sample = T.generate_trajectory(world, policy_exec, start, goal)
    #     # print("Start: %d, Policy:" %(start))
    #     D.append(sample)

    # print("\nTotal samples: %d\n" %(n_samples))
    # # Compute the robustness of goal-reach STTL
    # rho = sttl_goal_reach(D, goal)
    # print("StTL goal-reach robustness = %.2f\n" % (rho))

    # # Compute the robustness of obstacle avoidance STTL
    # rho = sttl_obstacle_avoid(D, goal, avoid_states)
    # print("StTL obstacle-avoidance robustness = %.2f\n" % (rho))

if __name__ == "__main__":
    main()