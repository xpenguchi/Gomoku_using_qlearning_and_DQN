#%%
import tkinter as tk
from gomokuenv.base_env import gomokuEnv




#%%
# define the UI of gomuku
class gomuku_UI:
  def __init__(self, board_size=15, cell_size=40):
    self.board_size = board_size
    self.cell_size = cell_size
    self.padding = 20
    # create the window
    self.root = tk.Tk()
    self.root.title("Gomoku using RL")
    # create the canvas
    canvas_size = board_size * cell_size + 2 * self.padding
    self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="#CDBA96")
    self.canvas.pack()
    self._draw_board()

  def _draw_board(self):
    # draw the board
    for i in range(self.board_size):
      # vertical lines
      x = self.padding + i * self.cell_size
      self.canvas.create_line(x, self.padding,
                              x, self.padding + self.board_size * self.cell_size,
                              fill="black")
      # horizontal lines
      y = self.padding + i * self.cell_size
      self.canvas.create_line(self.padding, y,
                              self.padding + self.board_size * self.cell_size, y,
                              fill="black")

  def draw_stone(self, row, col, player):
      # define the color
      color = "black" if player == 1 else "white"
      # draw the stone
      x = self.padding + col * self.cell_size
      y = self.padding + row * self.cell_size
      self.canvas.create_oval(x - self.cell_size // 2 + 3, y - self.cell_size // 2 + 3,
                              x + self.cell_size // 2 - 3, y + self.cell_size // 2 - 3,
                              fill=color,
                              outline = 'gray')
      self.root.update()

  def clear_board(self):
      self.canvas.delete("all")
      self._draw_board()
      self.root.update()

#%%
def visualization_gomuku(agent1, agent2, board_size=15):
  ui = gomuku_UI(board_size=board_size)
  env = gomokuEnv(board_size=board_size, ui=ui)
  state = env.reset()
  done = False
  agent = agent1
  while not done:
    # get the action space
    legal_actions = [i*board_size + j
                     for i in range(board_size)
                     for j in range(board_size)
                     if state[i, j] == 0]
    if len(legal_actions) == 0:
      print("Game Over. Tie")
      break
    # choose one action from the action space
    action = agent.choose_action(state, legal_actions)
    # operation chosen action
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
      if reward > 0:
        print("Game Over. Agent Win.")
      else:
        print("Game Over. Draw or Loss.")
    if agent == agent1:
      agent = agent2
    elif agent == agent2:
      agent = agent1
    
    ui.root.mainloop()



#%%
def play_two_agents(agent1, agent2, board_size=15):
    """Let two agents play with each other"""
    ui = gomuku_UI(board_size=board_size)
    
    env = gomokuEnv(board_size=board_size, ui=ui)
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
            print("Draw!")
            break
        
        current_agent = agent1 if turn % 2 == 0 else agent2
        action = current_agent.get_action(state, legal_actions)
        state, reward, done, _ = env.step(action)
        
        if done and reward > 0:
            winner = "Black" if turn % 2 == 0 else "White"
            print(f"Game Over, {winner} Wins!")
            break
            
        turn += 1
    
    # hold the window of ui opening
    ui.root.mainloop()


