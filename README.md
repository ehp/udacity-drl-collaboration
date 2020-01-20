# Udacity deep reinforcement learning - collaboration and competition

## Introduction

This project train two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.
The environment is considered solved, when the score average (over 100 episodes) is at least +0.5.

* Link to [original repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).
* Link to [training reports](Report.md).

## Environment details

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket.
Each agent receives its own, local observation. 
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 
After each episode, add the rewards each agent received (without discounting). This yields 2 (potentially different) scores. Then take the maximum of these 2 scores.
This yields a single score for each episode.

## Installation

Exclusive virtualenv is recommended:

```bash
virtualenv --python /usr/bin/python3 .venv
. .venv/bin/activate
``` 

Install dependencies:
```bash
pip install -r requirements.txt
```

If you encounter missing tensorflow 1.7.1 dependency try this:
```bash
pip install torch numpy pillow matplotlib grpcio protobuf
pip install --no-dependencies unityagents
```

Download unity environment files:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

And/or [NoVis alternative](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) (only for Linux).

## Training

Run training with default parameters:

```bash
python3 training.py
```

All training parameters:

|Parameter|Description|Default value|
|---|---|---|
|--environment|Path to Unity environment files|Tennis_Linux_NoVis/Tennis.x86_64|
|--actor_model|Path to save actor model|checkpoint_actor.pth|
|--critic_model|Path to save critic model|checkpoint_critic.pth|
|--buffer|Replay buffer type - sample or prioritized|prioritized|
|--episodes|Maximum number of training episodes|2000|
|--frames|Maximum number of frames in training episode|1000|
|--target|Desired minimal average per 100 episodes|0.5|
|--buffer_size|Replay buffer size|1000000|
|--batch_size|Minibatch size|1024|
|--gamma|Discount factor|0.85|
|--tau|For soft update of target parameters|0.001|
|--alpha|Prioritized buffer - How much prioritization is used (0 - no prioritization, 1 - full prioritization)|0.5|
|--beta|Prioritized buffer - To what degree to use importance weights (0 - no corrections, 1 - full correction)|0.5|
|--actor_learning_rate|Learning rate|0.0005|
|--critic_learning_rate|Learning rate|0.0005|
|--cuda/--no_cuda|Force disable CUDA or autodetect|Autodetect|

## Testing

Run test with default parameters:

```bash
python3 testing.py
```

All testing parameters:

|Parameter|Description|Default value|
|---|---|---|
|--environment|Path to Unity environment files|Tennis_Linux/Tennis.x86_64|
|--actor_model|Path to save actor model|checkpoint_actor.pth|
|--critic_model|Path to save critic model|checkpoint_critic.pth|
|--cuda/--no_cuda|Force disable CUDA or autodetect|Autodetect|

Pretrained models for [actor](models/actor.pth) and [critic](models/critic.pth). 

## Future work

- Implement other RL algorithms like REINFORCE, TNPG, RWR, REPS, TRPO, CEM, CMA-ES etc..
- Use more hw/time to do full hyperparameter space search for the best hyperparameters of this task.

## Licensing

Code in this repository is licensed under the MIT license. See [LICENSE](LICENSE).
