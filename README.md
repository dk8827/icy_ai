# Icy Tower AI

This project is an implementation of a reinforcement learning agent that learns to play the game "Icy Tower". The agent is built using a Double Deep Q-Network (DDQN) with PyTorch and interacts with a custom game environment created with Pygame.

## Features

- **Play Manually**: You can play the game yourself using the keyboard.
- **AI Training**: Train a DDQN agent to play the game.
  - **UI Mode**: Watch the agent learn in real-time with a graphical interface.
  - **Headless Mode**: Train the agent without a UI for faster performance.
- **Watch the AI**: See the trained agent in action.
- **Progress Tracking**: Training progress, including scores, losses, and exploration rate (epsilon), is automatically saved as charts.
- **Saved Models**: The trained agent's model is saved and can be loaded for further training or playing.

## Requirements

The project requires the following Python libraries:

- `pygame`
- `torch`
- `numpy`
- `tqdm`
- `matplotlib`

## Installation

1.  Clone the repository to your local machine.
2.  Navigate to the project directory:
    ```bash
    cd icy_ai
    ```
3.  It's recommended to use a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
4.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the application, run the `program.py` script:

```bash
python program.py
```

This will open the main menu with the following options:

- **Play Game (Keyboard)**: Play Icy Tower yourself. Use the arrow keys to move and the spacebar or up arrow to jump.
- **Learn with UI**: Start training the AI with the Pygame UI enabled. This is slower but allows you to visualize the agent's behavior as it learns.
- **Learn without UI**: Run the training in headless mode. This is significantly faster and recommended for long training sessions. Progress will be printed to the console.
- **Play using AI**: Watch the best-performing trained agent play the game. A model must be trained first.

Training charts will be saved in the `charts/` directory, and the trained model will be saved as `models/icy_tower_ddqn.pth`.

## File Structure

- `program.py`: The main entry point of the application. It contains the main menu, the training loop, and functions for human/AI play.
- `game_logic.py`: Implements the core mechanics of the Icy Tower game, independent of any UI.
- `pygame_env.py`: A wrapper around the game logic that creates a Pygame-based environment compatible with the agent.
- `agent.py`: Contains the implementation of the DDQN agent, including the neural network model (using PyTorch) and the learning logic.
- `config.py`: Stores configuration variables such as screen dimensions, colors, and model paths.
- `requirements.txt`: A list of the Python packages required for the project.
- `models/`: The directory where the trained neural network models are saved.
- `charts/`: The directory where charts illustrating the training progress are saved. 