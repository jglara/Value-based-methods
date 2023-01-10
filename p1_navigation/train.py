from collections import deque
import numpy as np
import torch

def train_rl(env, agent, n_episodes=2000, max_t=1000, model_weights_file="checkpoint.pth", **params):
    """Train an agent with the environment and using hyper parameters
    
    Params
    ======
        env: gym environment to interact
        agent: agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        model_wegihts_file: file to save model weights
        params: hyper parameters
    """
    
    # get default brain
    brain_name = env.brain_names[0]

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 score
    eps = params["EPS_START"]                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(params["EPS_END"], params["EPS_DECAY"]*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_weights_file)
            break
    return scores