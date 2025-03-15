#%%
import gym
import numpy as np
import tkinter as tk
from gym import spaces
import random
from collections import defaultdict
import time



#%%
# define gomoku env using gym.Env
class gomokuEnv(gym.Env):
    def __init__(self, board_size=15, ui=None):
      super(gomokuEnv, self).__init__()
      # board
      self.board = None
      self.board_size = board_size
      # current player
      self.current_player = None
      # action space
      self.action_space = spaces.Discrete(board_size * board_size)
      # observation space (0: empty, 1:black, 2: white)
      self.observation_space = spaces.Box(low=-1, high=1, shape=(board_size, board_size), dtype=np.int8)
      # UI
      self.ui = ui

    # get the current state
    def _get_observation(self):
      return self.board.copy()
    
    # calculate the reward for each step
    def _calculate_reward(self, row, col):
        player = self.current_player
        threat_reward = self._check_threats(row, col, player)
        # check if this step can defense the attack
        defense_reward = self._check_threats(row, col, -player)
        # we encourage the agent to play in the central part
        center_reward = self._calculate_center_reward(row, col)
        # all reward is the sum of threat_reward, defense_reward and center_reward
        return threat_reward + defense_reward + center_reward

    # check if win or not
    def _check_win(self, row, col):
      player = self.board[row, col]
      directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
      for dr, dc in directions:
        count = 1
        # check by '+'
        r, c = row + dr, col + dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player):
          count += 1
          r, c = r + dr, c + dc
        # check by '-'
        r, c = row - dr, col - dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player):
          count += 1
          r, c = r - dr, c - dc
        if count >= 5:
          return True
      return False
    
    # check if there is a threat situation and give the corresponding reward
    def _check_threats(self, row, col, player):
      directions = [(1,0), (0,1), (1,1), (1,-1)]
      max_count = 0
      
      for dr, dc in directions:
        count = 1
        # check by '+'
        r, c = row + dr, col + dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r,c] == player):
          count += 1
          r, c = r + dr, c + dc
        # check by '-'
        r, c = row - dr, col - dc
        while (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r,c] == player):
          count += 1
          r, c = r - dr, c - dc   
        # update the max_count 
        max_count = max(max_count, count)
        # giving the reward per max_count
        if max_count == 5:
           return 100
        if max_count == 4:
            return 50  
        elif max_count == 3:
            return 30  
        elif max_count == 2:
            return 2   
        return 0
    
    # calculate the center reward
    def _calculate_center_reward(self, row, col):
        center = self.board_size // 2
        distance = abs(row - center) + abs(col - center)
        return max(0, 3*(self.board_size - distance))

    # operate one step
    def step(self, action):
      # get the row and col
      row, col = action // self.board_size, action % self.board_size
      # check whether the action is illegal or not:
      if self.board[row, col] != 0:
        # we use -60 reward as punishment
        return self._get_observation(), -60, True, {}
      # put the stone:
      self.board[row, col] = self.current_player
      if self.ui:
        self.ui.draw_stone(row, col, self.current_player)
        time.sleep(0.5)
      # check if win or not and calculate corresponding reward
      done = self._check_win(row, col)
      reward = self._calculate_reward(row, col)
      # switch the player
      self.current_player *= -1
      # output
      return self._get_observation(), reward, done, {}

    def reset(self):
      # reset the board
      self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
      # black goes first
      self.current_player = 1
      if self.ui:
        self.ui.clear_board()
      return self._get_observation()




#%%
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.2):
        """initialize Q-learning agent"""
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.lr = learning_rate 
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space = action_space
    
    def get_action(self, state, legal_actions):
        """choose an action"""
        state_key = self._get_state_key(state)
        
        # epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # get the action with max_q from legal actions
        q_values = self.q_table[state_key]
        legal_q_values = {action: q_values[action] for action in legal_actions}
        return max(legal_q_values.items(), key=lambda x: x[1])[0]
    
    def learn(self, state, action, reward, next_state, done):
        """learning to update the q value"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # get current q value
        old_q = self.q_table[state_key][action]
        
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_key])
        
        # update q value per Q-learning principle
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state_key][action] = new_q
    
    def _get_state_key(self, state):
        return state.tobytes()



#%%
def train_two_agents(episodes=1000, board_size=15):
    env = gomokuEnv(board_size=board_size)
    agent1 = QLearningAgent(env.action_space, learning_rate=0.1, epsilon=0.2)
    agent2 = QLearningAgent(env.action_space, learning_rate=0.1, epsilon=0.2)
    wins = {'agent1': 0, 'agent2': 0, 'draw': 0}
    
    print("Begin to train the model...")
    for episode in range(episodes):
        state = env.reset()
        done = False
        turn = 0
        
        while not done:
            legal_actions = [
                i * board_size + j 
                for i in range(board_size) 
                for j in range(board_size) 
                if state[i,j] == 0
            ]
            
            if not legal_actions:
                wins['draw'] += 1
                break
                
            current_agent = agent1 if turn % 2 == 0 else agent2
            action = current_agent.get_action(state, legal_actions)
            next_state, reward, done, _ = env.step(action)
            
            if done: #and reward > 0:
                if turn % 2 == 0:
                    print(reward)
                    wins['agent1'] += 1
                else:
                    print(reward)
                    wins['agent2'] += 1
            
            # learn and update the state
            current_agent.learn(state, action, reward, next_state, done)
            state = next_state
            turn += 1
        
        if episode % 100 == 0:
            print(f"{episode} Training Completed!")
            print(f"Current Record - Black: {wins['agent1']}, White: {wins['agent2']}, Draw: {wins['draw']}")
    
    return agent1, agent2