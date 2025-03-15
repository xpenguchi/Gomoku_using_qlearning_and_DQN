#%%
import numpy as np
import tensorflow as tf
import tkinter as tk
import time
import random
import os
from gomokuenv.base_env import gomokuEnv, QLearningAgent
from gomokuui.ui import gomuku_UI

class GomokuFeatureExtractor:
    """extract board features and shrink state spaces"""
    def __init__(self, board_size=15):
        self.board_size = board_size
        
    def extract_features(self, board, last_move=None):
        """extract key features
        Returns:
            numpy array: eigen vector
        """
        # 1. 3 channels: my_pieces, opponent_pieces and empty_positions
        my_pieces = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        opponent_pieces = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        empty_positions = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        current_player = 1  # assume current player is 1 (black)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == current_player:
                    my_pieces[i, j] = 1
                elif board[i, j] == -current_player:
                    opponent_pieces[i, j] = 1
                else:
                    empty_positions[i, j] = 1
        # stack all features as one multi-channel input
        features = np.stack([my_pieces, opponent_pieces, empty_positions], axis=-1)
        return features

class DQNAgentVisualizer:
    """DQN Agent Wrapper for Visualization"""
    def __init__(self, model_path, board_size=15, player_name="DQN agent"):
        self.board_size = board_size
        self.player_name = player_name
        self.feature_extractor = GomokuFeatureExtractor(board_size=board_size)
        
        try:
            # attempt to load the model
            print(f"Attempt to load the model from {model_path} .")
            self.model = tf.keras.models.load_model(model_path)
            print("Successfully loaded the model!")
        except Exception as e:
            print(f"Cannot load the model: {e} .")
            print("Creating temporary model for showing.")
            self.model = self._build_temp_model()
    
    def _build_temp_model(self):
        """creating temporary model for showing"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Flatten
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(self.board_size, self.board_size, 3)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.board_size * self.board_size, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def get_action(self, state, legal_actions):
        """Choose action: Use model to predict the best action"""
        features = self.feature_extractor.extract_features(state)
        q_values = self.model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
        legal_q_values = {action: q_values[action] for action in legal_actions}
        return max(legal_q_values.items(), key=lambda x: x[1])[0]

class SimpleAgent:
    """Simple heuristic agent, no pre-trained model required"""
    def __init__(self, board_size=15, player_name="Simple AI"):
        self.board_size = board_size
        self.player_name = player_name
    
    def get_action(self, state, legal_actions):
        """Select action: Use simple heuristic rules"""
        # 1. Make a move if a five-in-a-row can be formed
        for action in legal_actions:
            row, col = action // self.board_size, action % self.board_size
            # Temporarily simulate this step
            test_board = state.copy()
            test_board[row, col] = 1
            if self._check_win(test_board, row, col):
                return action
        
        # 2. Block if the opponent can form five in a row in the next move
        for action in legal_actions:
            row, col = action // self.board_size, action % self.board_size
            # Temporarily simulate opponet's this step
            test_board = state.copy()
            test_board[row, col] = -1
            if self._check_win(test_board, row, col):
                return action
        
        # 3. Prioritize positions near the center
        center = self.board_size // 2
        # Sort all valid actions by distance to the center
        sorted_actions = sorted(
            legal_actions,
            key=lambda a: abs(a // self.board_size - center) + abs(a % self.board_size - center)
        )
        
        # Randomly choose one from the positions closest to the center
        top_n = min(5, len(sorted_actions))
        return random.choice(sorted_actions[:top_n])
    
    def _check_win(self, board, row, col):
        player = board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # +
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 
                  0 <= c < self.board_size and 
                  board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            # -
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 
                  0 <= c < self.board_size and 
                  board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 5:
                return True
        
        return False

class HumanPlayer:
    """Human Player Interface"""
    def __init__(self, board_size=15, player_name="Human Player"):
        self.board_size = board_size
        self.player_name = player_name
        self.selected_action = None
        
    def get_action(self, state, legal_actions, ui):
        """Get the move from human input"""
        print(f"{self.player_name}'s turn. Please click on the board to select a move...")
        
        # set click event
        def on_click(event):
            # convert click coordinates to board position
            col = round((event.x - ui.padding) / ui.cell_size)
            row = round((event.y - ui.padding) / ui.cell_size)
            
            # check if it's inn leagal_actions
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                action = row * self.board_size + col
                if action in legal_actions:
                    self.selected_action = action
                    ui.root.quit()
                else:
                    print("Invaild Position! Reselect Please.")
        
        # bind the click event
        click_id = ui.canvas.bind("<Button-1>", on_click)
        
        # wait for human player's selection
        ui.root.mainloop()
        
        # unbind the click event
        ui.canvas.unbind("<Button-1>", click_id)
        
        return self.selected_action

class QLearningAdapter:
    """Adapter for Q-Learning agent"""
    def __init__(self, q_agent, board_size=15, player_name="Q-Learning agent"):
        self.q_agent = q_agent
        self.board_size = board_size
        self.player_name = player_name
        
    def get_action(self, state, legal_actions):
        """Choose action using Q-learning agent"""
        return self.q_agent.get_action(state, legal_actions)

class RandomAgent:
    """Completely random move agent for testing"""
    def __init__(self, board_size=15, player_name="Random AI"):
        self.board_size = board_size
        self.player_name = player_name
    
    def get_action(self, state, legal_actions):
        """Randomly select a valid action"""
        return random.choice(legal_actions)

def visualize_gameplay(agent1, agent2, board_size=15, delay=1.0):
    """Visualize the game between two agents"""
    # Create the visualization interface
    ui = gomuku_UI(board_size=board_size)
    
    # Create the gomuku env
    env = gomokuEnv(board_size=board_size, ui=ui)
    state = env.reset()
    done = False
    
    # Record the number of current turns
    turn = 0
    
    # display players' information
    player_info = tk.Label(ui.root, text=f"Black: {agent1.player_name}  |  White: {agent2.player_name}", font=("Arial", 12))
    player_info.pack(pady=5)
    
    # display current status
    status_label = tk.Label(ui.root, text="Game start, black goes first!", font=("Arial", 12))
    status_label.pack(pady=5)
    
    # add controol buttons
    control_frame = tk.Frame(ui.root)
    control_frame.pack(pady=10)
    
    # enable autoplay or not
    auto_play = tk.BooleanVar(value=True)
    auto_play_check = tk.Checkbutton(control_frame, text="Auto-Play", variable=auto_play)
    auto_play_check.pack(side=tk.LEFT, padx=10)
    
    # adjust delay time
    delay_var = tk.DoubleVar(value=delay)
    tk.Label(control_frame, text="Delay Time (s):").pack(side=tk.LEFT)
    delay_slider = tk.Scale(control_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, variable=delay_var)
    delay_slider.pack(side=tk.LEFT, padx=10)
    
    # create next_step_button
    next_step_button = tk.Button(control_frame, text="Next Step", 
                               command=lambda: ui.root.quit())
    next_step_button.pack(side=tk.LEFT, padx=10)
    
    # create restart_button
    restart_button = tk.Button(control_frame, text="Restart", 
                               command=lambda: restart_game())
    restart_button.pack(side=tk.LEFT, padx=10)
    
    def restart_game():
        nonlocal state, done, turn
        ui.clear_board()
        state = env.reset()
        done = False
        turn = 0
        status_label.config(text="Game restart, black goes first!")
        ui.root.quit()
    
    while True:
        if done:
            # Wait for user action after the game ends
            ui.root.mainloop()
            continue
            
        # get legal_actions
        legal_actions = [
            i * board_size + j 
            for i in range(board_size) 
            for j in range(board_size) 
            if state[i,j] == 0
        ]
        
        # If there are no valid moves left, the game ends
        if not legal_actions:
            status_label.config(text="The board is full, the game is a draw!")
            done = True
            continue
        
        # select current agent
        current_agent = agent1 if env.current_player == 1 else agent2
        player_mark = "Black" if env.current_player == 1 else "White"
        
        # update status_label
        status_label.config(text=f"Current Turn: {turn+1}, {player_mark}({current_agent.player_name}) is thinking...")
        ui.root.update()
        
        # add delay when autoplaying
        if auto_play.get() and not isinstance(current_agent, HumanPlayer):
            time.sleep(delay_var.get())
        
        # agent selecting
        if isinstance(current_agent, HumanPlayer):
            action = current_agent.get_action(state, legal_actions, ui)
        else:
            action = current_agent.get_action(state, legal_actions)
            # If not autoplay, wait for the user to click "Next Step"
            if not auto_play.get():
                ui.root.mainloop()
        
        # perform the action
        state, reward, done, _ = env.step(action)
        
        # check if done
        if done:
            winner = "Black" if env.current_player == -1 else "White"
            agent_name = agent1.player_name if winner == "Black" else agent2.player_name
            status_label.config(text=f"Game Over! {winner}({agent_name}) Win!")
            continue
            
        turn += 1

def find_latest_model(models_dir="models"):
    """find the latest model according to the models_dir"""
    if not os.path.exists(models_dir):
        print(f"Path to the model {models_dir} doesn't exist")
        return None
        
    final_model = os.path.join(models_dir, "gomoku_dqn_model_final.keras")
    if os.path.exists(final_model):
        return final_model
        
    latest_model = os.path.join(models_dir, "gomoku_dqn_model_latest.keras")
    if os.path.exists(latest_model):
        return latest_model
        
    all_models = []
    for item in os.listdir(models_dir):
        if item.startswith("gomoku_dqn_model_") and item.endswith(".keras"):
            try:
                episode = int(item[17:-6])
                all_models.append((os.path.join(models_dir, item), episode))
            except ValueError:
                continue
            
    if all_models:
        all_models.sort(key=lambda x: x[1])
        return all_models[-1][0]  # find the latest model by its key

    return None

def train_quick_qlearning(episodes=500, board_size=15):
    """quickly train a new Q-Learning agent"""
    env = gomokuEnv(board_size=board_size)
    agent1 = QLearningAgent(env.action_space, learning_rate=0.1, epsilon=0.2)
    agent2 = QLearningAgent(env.action_space, learning_rate=0.1, epsilon=0.2)
    
    print("Training Q-Learning agent...")
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
                break
                
            current_agent = agent1 if turn % 2 == 0 else agent2
            action = current_agent.get_action(state, legal_actions)
            next_state, reward, done, _ = env.step(action)
            
            current_agent.learn(state, action, reward, next_state, done)
            state = next_state
            turn += 1
        
        if episode % 100 == 0:
            print(f"Training {episode}/{episodes} ...")
    
    print("Training completed!")
    return agent1, agent2

def human_vs_ai(model_path=None, board_size=15, human_first=True, use_simple_ai=False):
    
    if model_path is None or not os.path.exists(model_path):
        model_path = find_latest_model()
        print(f"Using latest model: {model_path}")
        
    if use_simple_ai or model_path is None:
        print("Using Simple AI")
        ai_agent = SimpleAgent(board_size=board_size, player_name="Simple AI")
    else:
        print(f"Using the DQN model: {model_path}")
        ai_agent = DQNAgentVisualizer(
            model_path=model_path,
            board_size=board_size,
            player_name="DQN Agent"
        )
    
    human_player = HumanPlayer(board_size=board_size)
    
    if human_first:
        black_player = human_player
        white_player = ai_agent
    else:
        black_player = ai_agent
        white_player = human_player
    
    visualize_gameplay(black_player, white_player, board_size=board_size)

def ai_vs_ai(model_path1=None, model_path2=None, board_size=15, use_simple_ai=False):
    """visualize the match between ai and ai"""
    
    if model_path1 is None or not os.path.exists(model_path1):
        model_path1 = find_latest_model()
        print(f"AI 1 using the latest model: {model_path1}")
        
    if use_simple_ai or model_path1 is None:
        print("AI 1 using Simple Agent")
        ai1 = SimpleAgent(board_size=board_size, player_name="Simple AI-1")
    else:
        print(f"AI 1 Using DQN model: {model_path1}")
        ai1 = DQNAgentVisualizer(
            model_path=model_path1,
            board_size=board_size,
            player_name="DQN Agent-1"
        )
    
    if model_path2 is None and not use_simple_ai:
        print("AI 2 uses the simple AI model")
        ai2 = SimpleAgent(board_size=board_size, player_name="Simple AI-2")
    elif model_path2 is not None:
        print(f"AI 2 uses DQN model: {model_path2}")
        ai2 = DQNAgentVisualizer(
            model_path=model_path2,
            board_size=board_size,
            player_name="DQN Agent-2"
        )
    else:
        print("AI 2 uses the random AI model")
        ai2 = RandomAgent(board_size=board_size, player_name="Random AI")
    
    visualize_gameplay(ai1, ai2, board_size=board_size)

#%%
if __name__ == "__main__":
    board_size = 15
    models_dir = "models"
    
    print("Please select a game mode:")
    print("1. Human vs DQN AI (if a trained model is available)")
    print("2. Human vs Simple AI (rule-based AI)")
    print("3. Human vs Random AI")
    print("4. DQN AI vs Simple AI (if a trained model is available)")
    print("5. Human vs Fast-trained Q-Learning AI")
    
    choice = input("Please enter the option from (1-5): ")
    
    latest_model = find_latest_model(models_dir)
    # latest_model = os.path.join(models_dir, "gomoku_dqn_model_100.keras")
    if latest_model:
        print(f"Find the trained model: {latest_model}")
    else:
        print("Cannot find the trained model.")
    
    if choice == "1":
        human_vs_ai(model_path=latest_model, board_size=board_size, human_first=True)
    elif choice == "2":
        human_vs_ai(board_size=board_size, human_first=True, use_simple_ai=True)
    elif choice == "3":
        human_player = HumanPlayer(board_size=board_size)
        random_ai = RandomAgent(board_size=board_size)
        visualize_gameplay(human_player, random_ai, board_size=board_size)
    elif choice == "4":
        ai_vs_ai(model_path1=latest_model, board_size=board_size)
    elif choice == "5":
        print("Training Q-Learning agent...")
        q_agent1, q_agent2 = train_quick_qlearning(episodes=500, board_size=board_size)
        human_player = HumanPlayer(board_size=board_size)
        q_adapter = QLearningAdapter(q_agent1, board_size=board_size)
        visualize_gameplay(human_player, q_adapter, board_size=board_size)
    else:
        print("Invalid option, defaulting to Human vs Simple AI mode.")
        human_vs_ai(board_size=board_size, human_first=True, use_simple_ai=True)
# %%
