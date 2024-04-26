"""
    Vanilly Policy Gradient (VPG) implementation.
    Based on Spinning Up's VPG implementation.
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from torch.optim import Adam
from torch.distributions.categorical import Categorical

from gymnasium.spaces import Discrete, Box

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """Build a FeedForward Neural Network"""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', 
          hidden_sizes=[32],
          lr=1e-2,
          epochs=50,
          batch_size=5000,
          render=False,
          ):
    
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name, render_mode="human")
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."
    
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss functinwhose gradient, for the right data, is policy gradient    
    def compute_loss(obs, act, weights):
        """Pseudo-loss function to adjust policy network weights (estimate gradients)"""
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make function to compute rewards on-to-go
    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs
    
    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []     # for measuring episode returns
        batch_lens = []     # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()[0]    # first obs comes from starting distribution
        done = False            # signal from environment that episode is over  
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if not finished_rendering_this_epoch and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32), 
            act=torch.as_tensor(batch_acts, dtype=torch.int64), 
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
            )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    logging.info("Training VPG on %s: %s", env_name, env)
    for i in range(epochs):
        logging.info("Starting epoch %s", i + 1)
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        logging.info('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    logging.info("Using simples formulation of policy gradient.")
    logging.info("Using arguments: %s", args)
    train(env_name=args.env_name, render=args.render, lr=args.lr)
