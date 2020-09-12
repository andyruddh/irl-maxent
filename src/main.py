#!/usr/bin/env python

import gridworld as W
import maxent as M
import plot as P
import trajectory as T
import solver as S
import optimizer as O
import mce_irl as I

import numpy as np
import matplotlib.pyplot as plt
import os

def get_policy(start):
    pass

def sttl_goal_reach():
    '''
        F (Pr[pos = goal] > 0.99)
    '''
    pass

def sttl_obstacle_avoid():
    '''
        G (Pr[dist2obs > 2] > 0.99)
    '''
    pass

def main():
    reward_mce, world, goal = I.mce_irl()
    value = S.value_iteration(world.p_transition, reward_mce, discount = 0.7)
    weighting = lambda x: x**5
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)

    # print("Value\n", value)
    # print("Policy\n", policy)
    # print("Policy Exec\n", policy_exec)

    D = []
    np.random.seed(0)
    for _ in range(3):
        start = np.random.randint(0, 16)
        sample = T.generate_trajectory(world, policy_exec, start, goal)
        # print("Start: %d, Policy:" %(start))
        D.append(sample)


if __name__ == "__main__":
    main()