This is our final project of DATA 37200 in uchicago.

This project explores the application of reinforcement learning techniques, specifi-
cally Q-learning and Deep Q-Networks (DQN), to develop an AI agent capable of
playing Gomoku (Five-in-a-Row). Traditional Q-learning struggles with large state
spaces like those found in board games, prompting our investigation into DQN as a
more scalable solution.

Our implementation creates a custom Gomoku environment using OpenAI Gym
and visualizes gameplay through Tkinter. The reward system incorporates threat
detection, defensive play assessment, and positional strategy to guide the learning
process. We compare several AI agents including a DQN-trained model, a simple
rule-based AI, a basic Q-learning implementation, and a random player through
five different game modes.

Training our DQN model required significant computational resources, taking
approximately 21,239 seconds to complete 500 episodes with periodic model
saving. While the model demonstrated learning progress with win rates fluctuating
between 45-58% during training, our preliminary results suggest challenges with
convergence, possibly due to overfitting or the complexity of the state space.
These findings highlight important considerations for applying deep reinforcement
learning to complex board games and provide a foundation for future optimization
efforts.

The ui of this project is designed like this:

https://github.com/user-attachments/assets/5f794927-8e9c-4961-b025-2ed5ac623ecc

You can play with it if you are interested! We also note that our ai agents' performance 
is not that perfect, and if you have ideas to refine our algo, feel free to email [me](xpengzhang0302@uchicago.edu). 


