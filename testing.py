from unityagents import UnityEnvironment
import torch
import argparse
import numpy as np

from agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', type=str, help='Path to Unity environment files',
                        default='Tennis_Linux/Tennis.x86_64')
    parser.add_argument('--actor_model', type=str, help='Path to save actor model',
                        default='checkpoint_actor.pth')
    parser.add_argument('--critic_model', type=str, help='Path to save critic model',
                        default='checkpoint_critic.pth')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    print('Testing')
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # initialize agent
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0,
                  num_agents=num_agents, training=False, args=args)
    # load nn
    agent.actor_local.load_state_dict(torch.load(args.actor_model, map_location=lambda storage, loc: storage))

    # play game for 5 episodes
    for i in range(1, 6):
        # reset the environment
        env_info = env.reset(train_mode=False)[brain_name]

        state = env_info.vector_observations[0]            # get the current state
        scores = np.zeros(num_agents)                      # initialize the score (for each agent)
        while True:
            # TODO remove actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = agent.act(state)                      # select an action
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break

        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

    env.close()
