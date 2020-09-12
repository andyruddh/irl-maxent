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


def main():
    reward_mce, world = I.mce_irl()
    value = S.value_iteration(world.p_transition, reward_mce, discount = 0.7)
    weighting = lambda x: x**5
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)

    # print(value)
    print(policy_exec)


if __name__ == "__main__":
    main()