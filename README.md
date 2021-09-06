# Implementation of  "Unsupervised Task Clustering for Multi-Task Reinforcement Learning", paper available [here](https://sites.google.com/view/unsupervised-task-clustering)

We use [`sacred`](https://github.com/IDSIA/sacred) to configure and monitor our experiments.

The implementation is split into three main files:
 * `train_em.py` for EM, SP and PPT
 * `train_multi_head.py` for the Multi-Head experiments
 * `train_atari_em.py` for all Atari experiments
 
 The implementation of the Atari experiments is closely based on the 
 [`Dopamine`](https://github.com/google/dopamine) framework.
 
### Usage

To use our code, please install dependencies
```
pip install -r requirements.txt
```
and run our experiments with
```
python train_em.py
```

To chose a different set of tasks, use the sacred CLI logic, for example
to replicate our experiments on the pendulum task set use:
```
python train_em.py with environment=pendulum
```
Logging is done via the sacred logging interface, which stores the results
in `results/sacred/$EXPERIMENT_ID/`. They can then be processed for example
with [`incense`](https://github.com/JarnoRFB/incense).

### Potential Issues
If the source code is not running as expected, please try to install some system packages that `gym`
might require for the atari games.
```
apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
```
