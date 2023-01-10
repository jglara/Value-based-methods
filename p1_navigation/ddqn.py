import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from buffer import ReplayBuffer

class DuelingQNetwork(nn.Module):
    """Dueling Q network Model."""

    def __init__(self, state_size, hidden_1, hidden_2, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.feature_model = nn.Sequential(nn.Linear(state_size, hidden_1), nn.ReLU())

        self.state_value_model = nn.Sequential(nn.Linear(hidden_1, hidden_2), nn.ReLU(),
                                               nn.Linear(hidden_2,1))
        
        self.advantage_model = nn.Sequential(nn.Linear(hidden_1, hidden_2), nn.ReLU(),
                                             nn.Linear(hidden_2, action_size))
      

    def forward(self, state):
        """Build a network that maps state -> action values."""
        fs = self.feature_model(state)
        vs = self.state_value_model(fs)
        advs = self.advantage_model(fs)

        return vs + advs - advs.mean()


class DuelingDQNAgent():
    """Dueling DQN Agent implementation"""

    def __init__(self, state_size, action_size, seed, device, **hparam):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            device: device to run model
            hparam: dictionary with hyper parameters
           
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.hparam = hparam
        self.device = device

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, hparam["HIDDEN_1"], hparam["HIDDEN_2"], action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, hparam["HIDDEN_1"], hparam["HIDDEN_2"], action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hparam["LR"])

        # Replay memory
        self.memory = ReplayBuffer(action_size, hparam["BUFFER_SIZE"], hparam["BATCH_SIZE"], seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.hparam["UPDATE_EVERY"]
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.hparam["BATCH_SIZE"]:
                experiences = self.memory.sample(self.device)
                self.learn(experiences, self.hparam["GAMMA"])

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## compute and minimize the loss
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.hparam["TAU"])         

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


