"""
Author:         Erfan Azad (basiri@cs.uni-freiburg.de)
Date:           3 December 2020
Description:    This is a file in with we have extracted the MPC (controller) and CEM (optimizer from the rest of the code)
                in order to better understand the requirements of running the two as a separate unit inside other algorithms (e.g. SAC)
"""
import time
import numpy as np
from gym.envs.classic_control import PendulumEnv
from dotmap import DotMap
from MPC import MPC
from optimizers import CEMOptimizer
from dynamics_model import nn_constructor
from config.cartpole import CartpoleConfigModule
from env.cartpole import CartpoleEnv
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- setting up the environment ---
env = CartpoleEnv()
print(env.action_space)
print(env.observation_space)


def obs_cost_fn(obs):
    ee_pos = CartpoleConfigModule._get_ee_pos(obs)
    ee_pos -= CartpoleConfigModule.ee_sub
    ee_pos = ee_pos ** 2
    ee_pos = - ee_pos.sum(dim=1)
    return - (ee_pos / (0.6 ** 2)).exp()


def ac_cost_fn(acs):
    return 0.01 * (acs ** 2).sum(dim=1)


# --- setting up the MPC and CEM optimizer ---
mpc_params = DotMap(
    env=env,
    prop_cfg=DotMap(
        model_init_cfg=DotMap(
            num_nets=5,
            input_dim=6,  # 5 obs + 1 ac
            output_dim=4,  # 4 obs???
            model_constructor=nn_constructor,
        ),
        model_train_cfg=DotMap(
            epochs=5,
        ),
        obs_preproc=CartpoleConfigModule.obs_preproc,
        obs_postproc=CartpoleConfigModule.obs_postproc,
        targ_proc=CartpoleConfigModule.targ_proc,
        model_pretrained=False,
        npart=20,
        ign_var=False,
        mode='TSinf',
    ),
    opt_cfg=DotMap(
        mode='CEM',
        plan_hor=20,
        obs_cost_fn=obs_cost_fn,
        ac_cost_fn=ac_cost_fn,
        cfg=DotMap(
            alpha=0.1,
            max_iters=5,
            num_elites=40,
            popsize=100,
        ),
    ),
)


def create_dataset(size, env):
    mpc = MPC(mpc_params)
    mpc.reset()
    O, A, R, done = [env.reset()], [], [], False
    for i in tqdm(range(size)):
        obs = O[i]
        action = mpc.act(obs)
        obs_next, reward, done, info = env.step(action)
        A.append(action)
        O.append(obs_next)
        R.append(reward)
        if done:
            print(f'Done at step: {i}')
            break
    return np.array(O), np.array(A), np.array(R)


def run_experiment(nepisodes, nsteps, env, mpc):
    nepisodes = nepisodes
    nsteps = nsteps
    episode_rewards = []
    for i in tqdm(range(nepisodes)):
        rewards = []
        obs = env.reset()
        for k in tqdm(range(nsteps)):
            action = mpc.act(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                print(f'Test episode done in {k} steps')
                break
        episode_rewards.append(rewards)
    return episode_rewards


# =============================
# === Testing Multiple Runs ===
# =============================
if(False):
    trainsize = 5000
    nteststeps = 100
    ntestepisodes = 3

    # Default Setting (or change here)
    mpc = MPC(mpc_params)
    observations, actions, rewards = create_dataset(size=trainsize, env=env)
    mpc.train([observations], [actions], [rewards])  # rewards is not used
    t0 = time.time()
    episode_rewards = run_experiment(
        nepisodes=ntestepisodes, nsteps=nteststeps, env=env, mpc=mpc)
    t1 = time.time()
    fig, ax = plt.subplots()
    for i in range(ntestepisodes):
        ax.plot(np.cumsum(episode_rewards[i]))
    ax.set_title(
        f'Trajectory rewards | trainsize:{trainsize} | Action Wallclock: {(t1 - t0)/(ntestepisodes*nteststeps):.2f}s')
    ax.set_xlabel('Test trajectory steps')
    ax.set_ylabel('Cumulative Reward')
    plt.show()
    fig.savefig(f'rewards_multiple_runs_{trainsize}train')

# =============================
# === Testing Effect of Number of Training Samples ===
# =============================
if(False):
    nteststeps = 200
    ntrain = [5000, 1000, 500, 50, 5]
    total_rewards = []
    total_time = []

    for n in ntrain:
        observations, actions, rewards = create_dataset(
            size=n, env=env)
        # Default Setting (or change here)
        mpc = MPC(mpc_params)
        mpc.train([observations], [actions], [rewards])  # rewards is not used
        t0 = time.time()
        episode_rewards = run_experiment(
            nepisodes=1, nsteps=nteststeps, env=env, mpc=mpc)
        t1 = time.time()
        total_time.append(t1 - t0)
        total_rewards.extend(episode_rewards)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in range(len(total_rewards)):
        ax1.plot(
            total_rewards[i], label=f'n:{ntrain[i]}|{total_time[i]/nteststeps:.2f}s')
        ax2.plot(
            np.cumsum(total_rewards[i]), label=f'n:{ntrain[i]}|{total_time[i]/nteststeps:.2f}s')
    # instant reward
    ax1.set_title(
        f'Trajectory rewards (Instant) | Training Samples Effect | Wallclock / Action')
    ax1.set_xlabel('Test trajectory steps')
    ax1.set_ylabel('Instant Reward')
    ax1.legend()
    # cumsum reward
    ax2.set_title(
        f'Trajectory rewards (Cumulative) | Training Samples Effect | Wallclock / Action')
    ax2.set_xlabel('Test trajectory steps')
    ax2.set_ylabel('Cumulative Reward')
    ax2.legend()
    fig.tight_layout(w_pad=3)
    plt.show()
    fig.savefig(f'results/rewards_ntrain')


# =============================
# === Testing Effect of Planning Horizon ===
# =============================
if(False):
    trainsize = 5000
    nteststeps = 200
    plan_hor = [30, 20, 10, 5, 2]
    total_rewards = []
    total_time = []

    observations, actions, rewards = create_dataset(size=trainsize, env=env)
    for hor in plan_hor:
        # Default Setting (or change here)
        mpc_params.opt_cfg.plan_hor = hor
        mpc = MPC(mpc_params)
        mpc.train([observations], [actions], [rewards])  # rewards is not used
        t0 = time.time()
        episode_rewards = run_experiment(
            nepisodes=1, nsteps=nteststeps, env=env, mpc=mpc)
        t1 = time.time()
        total_time.append(t1 - t0)
        total_rewards.extend(episode_rewards)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in range(len(total_rewards)):
        ax1.plot(
            total_rewards[i], label=f'horizon:{plan_hor[i]}|{total_time[i]/nteststeps:.2f}s')
        ax2.plot(
            np.cumsum(total_rewards[i]), label=f'horizon:{plan_hor[i]}|{total_time[i]/nteststeps:.2f}s')
    # instant reward
    ax1.set_title(
        f'Trajectory rewards (Instant) | Planning Horizon Effect | Wallclock / Action')
    ax1.set_xlabel('Test trajectory steps')
    ax1.set_ylabel('Instant Reward')
    ax1.legend()
    # cumsum reward
    ax2.set_title(
        f'Trajectory rewards (Cumulative) | Planning Horizon Effect | Wallclock / Action')
    ax2.set_xlabel('Test trajectory steps')
    ax2.set_ylabel('Cumulative Reward')
    ax2.legend()
    fig.tight_layout(w_pad=3)
    plt.show()
    fig.savefig(f'results/rewards_plan_hor')


# =============================
# === Testing Effect of Number of Particles ===
# =============================
if(False):
    trainsize = 5000
    nteststeps = 200
    nparticles = [200, 100, 50, 20, 10]
    total_rewards = []
    total_time = []

    observations, actions, rewards = create_dataset(size=trainsize, env=env)
    for n in nparticles:
        # Default Setting (or change here)
        mpc_params.opt_cfg.cfg.popsize = n
        mpc_params.opt_cfg.cfg.num_elites = int(n*0.3)  # 30% of pop
        mpc = MPC(mpc_params)
        mpc.train([observations], [actions], [rewards])  # rewards is not used
        t0 = time.time()
        episode_rewards = run_experiment(
            nepisodes=1, nsteps=nteststeps, env=env, mpc=mpc)
        t1 = time.time()
        total_time.append(t1 - t0)
        total_rewards.extend(episode_rewards)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in range(len(total_rewards)):
        ax1.plot(
            total_rewards[i], label=f'n:{nparticles[i]}|{total_time[i]/nteststeps:.2f}s')
        ax2.plot(
            np.cumsum(total_rewards[i]), label=f'n:{nparticles[i]}|{total_time[i]/nteststeps:.2f}s')
    # instant reward
    ax1.set_title(
        f'Trajectory rewards (Instant) | NParticles Effect | Wallclock / Action')
    ax1.set_xlabel('Test trajectory steps')
    ax1.set_ylabel('Instant Reward')
    ax1.legend()
    # cumsum reward
    ax2.set_title(
        f'Trajectory rewards (Cumulative) | NParticles Effect | Wallclock / Action')
    ax2.set_xlabel('Test trajectory steps')
    ax2.set_ylabel('Cumulative Reward')
    ax2.legend()
    fig.tight_layout(w_pad=3)
    plt.show()
    fig.savefig(f'results/rewards_nparticles')

# =============================
# === Testing Effect of Max_iter for the CEM optimizer ===
# =============================
if(False):
    trainsize = 5000
    nteststeps = 200
    max_iters = [10, 5, 3, 2, 1]
    total_rewards = []
    total_time = []

    observations, actions, rewards = create_dataset(size=trainsize, env=env)
    for n in max_iters:
        # Default Setting (or change here)
        mpc_params.opt_cfg.cfg.max_iters = n
        mpc = MPC(mpc_params)
        mpc.train([observations], [actions], [rewards])  # rewards is not used
        t0 = time.time()
        episode_rewards = run_experiment(
            nepisodes=1, nsteps=nteststeps, env=env, mpc=mpc)
        t1 = time.time()
        total_time.append(t1 - t0)
        total_rewards.extend(episode_rewards)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in range(len(total_rewards)):
        ax1.plot(
            total_rewards[i], label=f'max_iter:{max_iters[i]}|{total_time[i]/nteststeps:.2f}s')
        ax2.plot(
            np.cumsum(total_rewards[i]), label=f'max_iter:{max_iters[i]}|{total_time[i]/nteststeps:.2f}s')
    # instant reward
    ax1.set_title(
        f'Trajectory rewards (Instant) | CEM Max_iter Effect | Wallclock / Action')
    ax1.set_xlabel('Test trajectory steps')
    ax1.set_ylabel('Instant Reward')
    ax1.legend()
    # cumsum reward
    ax2.set_title(
        f'Trajectory rewards (Cumulative) | CEM Max_iter Effect | Wallclock / Action')
    ax2.set_xlabel('Test trajectory steps')
    ax2.set_ylabel('Cumulative Reward')
    ax2.legend()
    fig.tight_layout(w_pad=3)
    plt.show()
    fig.savefig(f'results/rewards_CEM_max_iters')


# =============================
# === Testing Resulting Configuration ===
# =============================
if(True):
    trainsize = 20
    nteststeps = 2
    nepisodes = 3

    total_rewards = []
    total_time = 0.0

    observations, actions, rewards = create_dataset(size=trainsize, env=env)
    # Default Setting (or change here)
    mpc_params.opt_cfg.cfg.max_iters = 3
    mpc_params.opt_cfg.cfg.popsize = 50
    mpc_params.opt_cfg.cfg.num_elites = 10
    mpc_params.opt_cfg.plan_hor = 15

    mpc = MPC(mpc_params)
    mpc.train([observations], [actions], [rewards])  # rewards is not used
    t0 = time.time()
    episode_rewards = run_experiment(
        nepisodes=nepisodes, nsteps=nteststeps, env=env, mpc=mpc)
    t1 = time.time()
    total_time = t1 - t0
    total_rewards.extend(episode_rewards)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in range(len(total_rewards)):
        ax1.plot(
            total_rewards[i], label=f'Wallclock/Action:{total_time/(nepisodes*nteststeps):.2f}s')
        ax2.plot(
            np.cumsum(total_rewards[i]), label=f'Wallclock/Action:{total_time/(nepisodes*nteststeps):.2f}s')
    # instant reward
    ax1.set_title(
        f'Trajectory rewards (Instant) | Optimized | Wallclock / Action')
    ax1.set_xlabel('Test trajectory steps')
    ax1.set_ylabel('Instant Reward')
    ax1.legend()
    # cumsum reward
    ax2.set_title(
        f'Trajectory rewards (Cumulative) | Optimized | Wallclock / Action')
    ax2.set_xlabel('Test trajectory steps')
    ax2.set_ylabel('Cumulative Reward')
    ax2.legend()
    fig.tight_layout(w_pad=3)
    plt.show()
    fig.savefig(f'results/rewards_optimized_test')
