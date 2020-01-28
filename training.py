import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from agent import Agent


# states 4,12 and 20 are centered around 0
def flip_state(state):
    state[4] = -1 * state[4]
    state[12] = -1 * state[12]
    state[20] = -1 * state[20]

    return state


def ppo(n_episodes=2000, max_t=1000, writer=None, agent=None,
        target=100.0, policy_model='checkpoint_policy.pth'):
    """Proximal Policy Optimization.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        num_agents (int): number of agents
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        target (float): desired minimal average per 100 episodes
        policy_model (str): path to save policy model
    """
    all_scores = []
    scores_history = 100
    scores_buffer = np.zeros(scores_history)              # scores from each episode
    scores_window = 0                                     # circular buffer index
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros((num_agents))                      # scores for single episode
        for t in range(max_t):
            actions = np.zeros((num_agents, action_size))
            log_probs = np.zeros(num_agents)
            for idx, state in enumerate(states):
                # action 0: <-1,1> - right/left, action 1: >0.5 jump or nothing
                if idx % 2 == 1:
                    state = flip_state(state)

                actions[idx], log_probs[idx] = agent.act(state)
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished

            for idx, (state, action, reward, next_state, log_prob, done) in enumerate(zip(states, actions, rewards, next_states, log_probs, dones)):
                if idx % 2 == 1:
                    next_state = flip_state(next_state)

                agent.step(state, action, reward, next_state, log_prob, done)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        episode_score = np.max(scores)
        all_scores.append(episode_score)
        scores_buffer[scores_window] = episode_score
        if writer:
            writer.add_scalar('score', episode_score, i_episode)
            writer.add_scalar('frames', t, i_episode)
        scores_window = (scores_window + 1) % scores_history

        score_mean = np.mean(scores_buffer)
        if writer:
            writer.add_scalar('score_mean', score_mean, i_episode)
        print('\rEpisode {}\tAverage Score: {:.5f}'.format(i_episode, score_mean), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.5f}'.format(i_episode, score_mean))
        if score_mean >= target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.5f}'.format(i_episode - 100, score_mean))
            torch.save(agent.policy.state_dict(), agent.id + policy_model)
            break

    return all_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', type=str, help='Path to Unity environment files',
                        default='Tennis_Linux_NoVis/Tennis.x86_64')
    parser.add_argument('--policy_model', type=str, help='Path to save policy model',
                        default='checkpoint_actor.pth')
    parser.add_argument('--buffer', type=str, help='Replay buffer type - sample or prioritized',
                        default='prioritized')
    parser.add_argument('--episodes', type=int, help='Maximum number of training episodes',
                        default=20000)
    parser.add_argument('--frames', type=int, help='Maximum number of frames in training episode',
                        default=1000)
    parser.add_argument('--target', type=float, help='Desired minimal average per 100 episodes',
                        default=0.5)
    parser.add_argument('--buffer_size', type=int, help='Replay buffer size',
                        default=102400)
    parser.add_argument('--batch_size', type=int, help='Minibatch size',
                        default=512)
    parser.add_argument('--gamma', type=float, help='Discount factor',
                        default=0.99)
    parser.add_argument('--eps_clip', type=float, help='PPO epsilon clip',
                        default=0.2)
    parser.add_argument('--c1', type=float, help='PPO VF coefficient',
                        default=0.5)
    parser.add_argument('--c2', type=float, help='PPO entropy coefficient',
                        default=0.01)
    parser.add_argument('--weight_decay', type=float, help='Optimizer weight decay',
                        default=0.0005)
    parser.add_argument('--tau', type=float, help='For soft update of target parameters',
                        default=0.1)
    parser.add_argument('--alpha', type=float, help='Prioritized buffer - How much prioritization is used (0 - no prioritization, 1 - full prioritization)',
                        default=0.5)
    parser.add_argument('--beta', type=float, help='Prioritized buffer - To what degree to use importance weights (0 - no corrections, 1 - full correction)',
                    default=0.5)
    parser.add_argument('--learning_rate', type=float, help='Learning rate',
                        default=0.0005)
    # actor lr 0.0001
    # critic lr 0.001
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    print('Training')
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # initialize agent
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    print('State size:', state_size)
    print('Action size:', action_size)

    writer = SummaryWriter()
    writer.add_hparams({'frames': args.frames,
                        'episodes': args.episodes,
                        'buffer_size': args.buffer_size,
                        'batch_size': args.batch_size,
                        'gamma': args.gamma,
                        'tau': args.tau,
                        'alpha': args.alpha,
                        'beta': args.beta,
                        'learning_rate': args.learning_rate,
                        'target': args.target}, {})

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, writer=writer,
                        id='PPO', training=True, args=args)

    scores = ppo(n_episodes=args.episodes, max_t=args.frames, target=args.target,
                 agent=agent, writer=writer, policy_model=args.policy_model)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(args.policy_model + '.png')

    env.close()
    writer.close()
