#%%
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from collections import deque
import random
import os
import time
from gomokuenv.base_env import gomokuEnv

class GomokuFeatureExtractor:
    def __init__(self, board_size=15):
        self.board_size = board_size
        
    def extract_features(self, board, last_move=None):
        my_pieces = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        opponent_pieces = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        empty_positions = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        current_player = 1
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == current_player:
                    my_pieces[i, j] = 1
                elif board[i, j] == -current_player:
                    opponent_pieces[i, j] = 1
                else:
                    empty_positions[i, j] = 1
        
        features = np.stack([my_pieces, opponent_pieces, empty_positions], axis=-1)
        return features

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target_model()
    
    def _build_model(self, learning_rate):
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.state_shape),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        loss_fn = MeanSquaredError()
        model.compile(loss=loss_fn, optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def get_action(self, state, legal_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        
        legal_q_values = {action: q_values[action] for action in legal_actions}
        return max(legal_q_values.items(), key=lambda x: x[1])[0]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def save(self, filepath):
        if not filepath.endswith('.keras'):
            filepath = filepath + '.keras'
            
        self.model.save(filepath)
        print(f"Model has benn saved in: {filepath}")
    
    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_model()
        print(f"Successfully loading the model from {filepath} !")

def train_dqn_agent(episodes=5000, board_size=15, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    env = gomokuEnv(board_size=board_size)
    
    feature_extractor = GomokuFeatureExtractor(board_size=board_size)
    
    sample_state = env.reset()
    sample_features = feature_extractor.extract_features(sample_state)
    state_shape = sample_features.shape
    
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=board_size * board_size,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01
    )
    
    wins = 0
    losses = 0
    draws = 0
    
    rewards_history = []
    loss_history = []
    
    print("Begin to train DQN agent...")
    start_time = time.time()
    
    for episode in range(1, episodes+1):
        state = env.reset()
        features = feature_extractor.extract_features(state)
        done = False
        total_reward = 0
        episode_loss = []
        
        while not done:
            legal_actions = [
                i * board_size + j 
                for i in range(board_size) 
                for j in range(board_size) 
                if state[i, j] == 0
            ]
            
            if not legal_actions:
                draws += 1
                break
                
            action = agent.get_action(features, legal_actions)
            
            next_state, reward, done, _ = env.step(action)
            next_features = feature_extractor.extract_features(next_state)
            
            agent.remember(features, action, reward, next_features, done)
            
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            total_reward += reward
            
            state = next_state
            features = next_features
        
        if done and total_reward > 0:
            if env.current_player == -1:
                wins += 1
            else:
                losses += 1
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        loss_history.append(avg_loss)
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            agent.update_target_model()
        
        if episode % 20 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(rewards_history[-20:])
            avg_loss = np.mean(loss_history[-20:])
            win_rate = wins / (wins + losses + draws) if (wins + losses + draws) > 0 else 0
            
            print(f"Episode: {episode}/{episodes} | "
                  f"Win Rate: {win_rate:.3f} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Avg Loss: {avg_loss:.6f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Time: {elapsed_time:.1f}s")
            
            if episode % 100 == 0:
                wins, losses, draws = 0, 0, 0
                
        if episode % 100 == 0 or episode == episodes:
            save_path = os.path.join(save_dir, f"gomoku_dqn_model_{episode}")
            agent.save(save_path)
            
            latest_path = os.path.join(save_dir, "gomoku_dqn_model_latest")
            agent.save(latest_path)
    
    final_path = os.path.join(save_dir, "gomoku_dqn_model_final")
    agent.save(final_path)
    
    print(f"Training Completed! Total Cost Time: {time.time() - start_time:.2f} seconds.")
    
    return agent, feature_extractor


#%%
if __name__ == "__main__":
    board_size = 15
    episodes = 500
    save_dir = "models"
    
    agent, feature_extractor = train_dqn_agent(
        episodes=episodes,
        board_size=board_size,
        save_dir=save_dir
    )
    
    print(f"Model has been save in {save_dir} .")
    print("Now we can use trained agent to perform visualization.")
# %%
