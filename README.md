# Trains an agent with Policy Gradient on Pong

The code is based on Andrej Karpathy's blog post ["Deep Reinforcement Learning: Pong from Pixels"](https://karpathy.github.io/2016/05/31/rl/).

There is a pickle file with trained weight to instantiate the policy around 6k episodes of gradient adjustments.

## Getting Started

Tested with `python 3.11.0`.

```bash
git clone git@github.com:alcazar90/pg-pong.git
python3 -m venv pong
source pong/bin/activate
pip install -r requirements
python pg-pong.py
```
