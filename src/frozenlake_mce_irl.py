#!/usr/bin/env python

import gridworld as W
import maxent as M
import plot as P
import trajectory as T
import solver as S
import optimizer as O

import numpy as np
import matplotlib.pyplot as plt
import os

def setup_mdp(GRID_SIZE, p_slip, avoid_states):
    """
    Set-up our MDP/GridWorld
    """
    # create our world
    world = W.IcyGridWorld(size=GRID_SIZE, p_slip=p_slip)

    # set up the reward function
    # reward = np.zeros(world.n_states)

    reward = np.ones(world.n_states)

    reward[3] = 2

    # Define some obstacles or avoid regions
    for s in avoid_states:
        reward[s] = 0

    # set up terminal states
    terminal = [3]
    # print(world.n_states)
    # print(reward)
    return world, reward, terminal


def generate_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 5
    print("\nNumber of experts: %d\n" %(n_trajectories))
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[12] = 1.0

    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

    return tjs, policy


def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward

def mce_irl(grid_size, p_slip, avoid_states):

    cwd = os.getcwd()
    fig_dir = "figs"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
   
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
        'cmap': "Blues",
    }

    # set-up mdp
    world, reward, terminal = setup_mdp(grid_size, p_slip, avoid_states)

    # show our original reward
    ax = plt.figure(num='Original Reward').add_subplot(111)
    tt = P.plot_state_values(ax, world, reward, **style)
    plt.colorbar(tt)
    plt.draw()

    plt.savefig(os.path.join(fig_dir, "gt_reward.png"))


    print("\nGenerating expert trajectories ...\n")
    # generate "expert" trajectories
    trajectories, expert_policy = generate_trajectories(world, reward, terminal)

    # show our expert policies
    ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    P.plot_stochastic_policy(ax, world, expert_policy, **style)

    for t in trajectories:
        P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    plt.draw()
    plt.savefig(os.path.join(fig_dir, "demonstrations.png"))


    '''
    print("ME-IRL ...")
    # maximum entropy reinforcement learning (non-causal)
    reward_maxent = maxent(world, terminal, trajectories)

    # show the computed reward
    ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    P.plot_state_values(ax, world, reward_maxent, **style)
    plt.draw()
    '''

    print("\nPerforming MCE-IRL ...\n")
    # maximum casal entropy reinforcement learning (non-causal)
    reward_maxcausal = maxent_causal(world, terminal, trajectories)

    # print(reward_maxcausal)

    # show the computed reward
    ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    tt = P.plot_state_values(ax, world, reward_maxcausal, **style)
    plt.colorbar(tt)
    plt.draw()
    plt.savefig(os.path.join(fig_dir, "maxent_causal_reward.png"))

    print("\nDone! Rewards learned\n")

    # plt.show()
    return (reward_maxcausal, world, terminal)