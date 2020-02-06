import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from agent import Agent


def ppo(n_episodes=2000, frames=1000, writer=None, agent=None, num_agents=2,
        target=100.0, policy_model='checkpoint_policy.pth', device='cpu'):
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
    i_episode = 0
    keep_loop = True
    states, scores = reset_episode(env, num_agents)
    while keep_loop:
        for t in range(frames):
            actions, log_probs, values = agent.act(states)
            env_actions = actions.clamp(min=-1.0, max=1.0).squeeze().detach().cpu().numpy()
            env_info = env.step(env_actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished
            scores += rewards
            end_episode = np.any(dones)

            next_states = torch.from_numpy(next_states).double().to(device)
            rewards = torch.tensor(rewards, device=device, dtype=torch.float64)
            dones = torch.tensor(dones, device=device, dtype=torch.int8)

            agent.save_step(actions, rewards, states, log_probs, values, dones)
            states = next_states
            if end_episode:
                # End episode
                episode_score = np.max(scores)
                all_scores.append(episode_score)
                score_mean = np.mean(all_scores[-100:])
                if writer:
                    writer.add_scalar('score', episode_score, i_episode)
                    writer.add_scalar('frames', t, i_episode)
                    writer.add_scalar('score_mean', score_mean, i_episode)
                print('\rEpisode {}\tEpisode score: {:.5f}\tAverage Score: {:.5f}'.format(i_episode, episode_score, score_mean), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tEpisode score: {:.5f}\tAverage Score: {:.5f}'.format(i_episode, episode_score, score_mean))
                if score_mean >= target:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.5f}'.format(i_episode - 100, score_mean))
                    torch.save(agent.policy.state_dict(), policy_model)
                    keep_loop = False
                    break
                if i_episode > n_episodes:
                    print('\nTarget not reached in %d episodes.' % n_episodes)
                    keep_loop = False
                    break

                i_episode += 1
                states, scores = reset_episode(env, num_agents)

        if keep_loop:
            agent.finish_rollout(states)

    return all_scores

def reset_episode(env, num_agents):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations
    states = torch.from_numpy(states).double().to(device)
    scores = np.zeros((num_agents))                   # scores for single episode

    return states, scores



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', type=str, help='Path to Unity environment files',
                        default='Tennis_Linux_NoVis/Tennis.x86_64')
    parser.add_argument('--policy_model', type=str, help='Path to save policy model',
                        default='checkpoint_policy.pth')
    parser.add_argument('--episodes', type=int, help='Maximum number of training episodes',
                        default=20000)
    parser.add_argument('--frames', type=int, help='Maximum number of frames in training episode',
                        default=64*4)
    parser.add_argument('--target', type=float, help='Desired minimal average per 100 episodes',
                        default=0.5)
    parser.add_argument('--batch_size', type=int, help='Minibatch size',
                        default=64)
    parser.add_argument('--training_epochs', type=int, help='How many times epoch is trained',
                        default=10)
    parser.add_argument('--gamma', type=float, help='Discount factor',
                        default=0.99)
    parser.add_argument('--gae', type=float, help='GAE coefficient',
                        default=0.95)
    parser.add_argument('--eps_clip', type=float, help='PPO epsilon clip',
                        default=0.1)
    parser.add_argument('--c1', type=float, help='PPO VF coefficient',
                        default=0.5)
    parser.add_argument('--c2', type=float, help='PPO entropy coefficient',
                        default=0.01)
    parser.add_argument('--weight_decay', type=float, help='Optimizer weight decay',
                        default=0.00001)
    parser.add_argument('--learning_rate', type=float, help='Learning rate',
                        default=0.0003)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    print('Training')
    args = parser.parse_args()

    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

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
                        'batch_size': args.batch_size,
                        'training_epochs': args.training_epochs,
                        'gamma': args.gamma,
                        'gae': args.gae,
                        'eps_clip': args.eps_clip,
                        'c1': args.c1,
                        'c2': args.c2,
                        'learning_rate': args.learning_rate,
                        'weight_decay': args.weight_decay,
                        'target': args.target}, {})

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, writer=writer,
                  num_agents=num_agents, training=True, args=args, device=device)

    scores = ppo(n_episodes=args.episodes, frames=args.frames, target=args.target,
                 device=device, num_agents=num_agents,
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
