# Policy Gradient Training for ATARI Pong Agent

This project trains an agent to play Pong using Policy Gradients. The implementation is based on Andrej Karpathy's code ["Training a Neural Network ATARI Pong agent with Policy Gradients from raw pixels
"](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5). For a detailed explanation of the code and deep reinforcement learning for policy function approximation, refer to the blog post ["Deep Reinforcement Learning: Pong from Pixels"](https://karpathy.github.io/2016/05/31/rl/).


A pickle file with trained weights is included, which can be used to instantiate the policy after approximately 6.4k + 2.9k episodes of gradient adjustments.

## Getting Started

The code has been tested with `Python 3.11.0`.

```bash
git clone git@github.com:alcazar90/pg-pong.git
python3 -m venv pong
source pong/bin/activate
pip install -r requirements
python pg-pong.py
```

This will start the training process for the Pong agent.


