from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from agent import Agent

def ppo(n_episodes=2000, max_t=1000, num_agents=2,
        target=100.0, actor_model='checkpoint_actor.pth', critic_model='checkpoint_critic.pth'):
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
        actor_model (str): path to save actor model
        critic_model (str): path to save critic model
    """
    scores_history = 100
    scores_buffer = np.zeros(scores_history)              # scores from each episode
    scores_window = 0                                     # circular buffer index
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)                      # scores for single episode
        for t in range(max_t):
            actions = agent.act(states)
            actions = np.clip(actions, -1, 1)              # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        scores_buffer[scores_window] = np.max(scores)
        scores_window = (scores_window + 1) % scores_history

        score_mean = np.mean(scores_buffer)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_mean), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_mean))
        if score_mean >= target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, score_mean))
            torch.save(agent.actor_local.state_dict(), actor_model)
            torch.save(agent.critic_local.state_dict(), critic_model)
            break

    # unroll scores
    return np.concatenate((scores_buffer[scores_window:], scores_buffer[:scores_window]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', type=str, help='Path to Unity environment files',
                        default='Tennis_Linux_NoVis/Tennis.x86_64')
    parser.add_argument('--actor_model', type=str, help='Path to save actor model',
                        default='checkpoint_actor.pth')
    parser.add_argument('--critic_model', type=str, help='Path to save critic model',
                        default='checkpoint_critic.pth')
    parser.add_argument('--buffer', type=str, help='Replay buffer type - sample or prioritized',
                        default='prioritized')
    parser.add_argument('--episodes', type=int, help='Maximum number of training episodes',
                        default=2000)
    parser.add_argument('--frames', type=int, help='Maximum number of frames in training episode',
                        default=1000)
    parser.add_argument('--target', type=float, help='Desired minimal average per 100 episodes',
                        default=0.5)
    parser.add_argument('--buffer_size', type=int, help='Replay buffer size',
                        default=1000000)
    parser.add_argument('--batch_size', type=int, help='Minibatch size',
                        default=1024)
    parser.add_argument('--gamma', type=float, help='Discount factor',
                        default=0.85)
    parser.add_argument('--tau', type=float, help='For soft update of target parameters',
                        default=0.001)
    parser.add_argument('--alpha', type=float, help='Prioritized buffer - How much prioritization is used (0 - no prioritization, 1 - full prioritization)',
                        default=0.5)
    parser.add_argument('--beta', type=float, help='Prioritized buffer - To what degree to use importance weights (0 - no corrections, 1 - full correction)',
                    default=0.5)
    parser.add_argument('--actor_learning_rate', type=float, help='Actor learning rate',
                        default=0.0005)
    parser.add_argument('--critic_learning_rate', type=float, help='Critic learning rate',
                        default=0.0005)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    print('Training')
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # number of agents
    num_agents = len(env.agents)
    print('Number of agents:', num_agents)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # initialize agent
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0,
                  num_agents=num_agents, training=True, args=args)

    scores = ppo(n_episodes=args.episodes, max_t=args.frames, target=args.target, num_agents=num_agents,
                  actor_model=args.actor_model, critic_model=args.critic_model)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(args.actor_model + '.png')

    env.close()
