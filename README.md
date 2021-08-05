
# Soft Actor Crtic Implementation

### Introduction
This is the Pytorch implementation of soft actor critic with auto temperature tuning.

### Getting Started

1. Install [mujoco-py](https://github.com/openai/mujoco-py) to test SAC in environment of `humanoid-v2`.

2. Download this repository.

3. If you want to test the saved model, then just set `is_train = False` in `main.py` or train by yourself then just set `is_train = True` in `main.py`
    a. If you want do parameter tuning, then modify parameters in `sac_agent.py`

### Plot

You can see by yourself reward, critic loss, actor loss, etc in `log` folder by tensorboard.

For convenient, I attached it.

![sac_humanoidv2](https://user-images.githubusercontent.com/73100569/128279001-b1a26517-848a-4ae4-ad17-735ccf064efa.png)