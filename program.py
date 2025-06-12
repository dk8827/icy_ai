import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque, namedtuple
import numpy as np
import os

# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================

# --- Game Configuration ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GAME_TITLE = "Icy Tower DDQN"
BACKGROUND_COLOR = (135, 206, 235) # Sky Blue
PLAYER_COLOR = (255, 0, 0) # Red
PLATFORM_COLOR = (100, 100, 100) # Gray

# --- Game Physics & Settings ---
PLAYER_WIDTH = 30
PLAYER_HEIGHT = 40
PLAYER_ACC = 0.8
PLAYER_FRICTION = -0.12
PLAYER_GRAVITY = 0.6
PLAYER_JUMP_POWER = -15

PLATFORM_MIN_WIDTH = 50
PLATFORM_MAX_WIDTH = 120
PLATFORM_HEIGHT = 20
PLATFORM_SPACING = 90

# --- DDQN Agent Hyperparameters ---
### CHANGE: Removed STATE_SIZE and ACTION_SIZE constants. They will be derived from the environment.
BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
LR = 5e-4               # Learning rate
TAU = 1e-3              # For soft update of target network
UPDATE_EVERY = 4        # How often to update the network

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define the Experience tuple for the Replay Buffer
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

# ==============================================================================
# 1. HYPERPARAMETERS AND CONFIGURATION
# ==============================================================================

# ==============================================================================
# 2. GAME COMPONENTS (PLAYER AND PLATFORM)
# ==============================================================================

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([PLAYER_WIDTH, PLAYER_HEIGHT])
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.pos = pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, PLAYER_GRAVITY)

    def move(self, action):
        # Action: 0=Left, 1=Stay, 2=Right
        self.acc.x = 0
        if action == 0:
            self.acc.x = -PLAYER_ACC
        if action == 2:
            self.acc.x = PLAYER_ACC

    def update(self):
        self.acc.x += self.vel.x * PLAYER_FRICTION
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        if self.pos.x > SCREEN_WIDTH:
            self.pos.x = 0
        if self.pos.x < 0:
            self.pos.x = SCREEN_WIDTH

        self.rect.midbottom = self.pos

    def jump(self):
        self.vel.y = PLAYER_JUMP_POWER

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width):
        super().__init__()
        self.image = pygame.Surface([width, PLATFORM_HEIGHT])
        self.image.fill(PLATFORM_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

# ==============================================================================
# 3. THE GAME ENVIRONMENT (Simplified Icy Tower)
# ==============================================================================

class IcyTowerEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(GAME_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.running = True

        ### NEW: Define the action space size within the environment itself.
        # This is a common practice (similar to OpenAI Gym).
        self.action_space_n = 3 # 0: Left, 1: Stay, 2: Right

    # --------------------------------------------------------------------------
    # Main Methods
    # --------------------------------------------------------------------------
    def reset(self):
        """Resets the environment to an initial state."""
        self.player = Player()
        self.player.pos = pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50)
        initial_platform = Platform(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH)
        self.platforms = pygame.sprite.Group(initial_platform)
        new_platforms = self._generate_platforms(15, SCREEN_HEIGHT - 100)
        self.platforms.add(new_platforms)
        self.all_sprites = pygame.sprite.Group(self.player, self.platforms)
        self.score = 0
        self.camera_y = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        """Processes an action, updates the game state, and returns results."""
        prev_score = self.score

        self._handle_player_movement(action)
        landed = self._handle_collisions()
        self._scroll_camera()
        self._manage_platforms()
        self._update_score()
        self._check_game_over()

        reward = self._calculate_reward(prev_score, landed)
        next_state = self._get_state()

        return next_state, reward, self.done, {}

    def render(self):
        """Draws the current game state to the screen."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

        self.screen.fill(BACKGROUND_COLOR)
        self.all_sprites.draw(self.screen)
        score_text = self.font.render(f"Score: {self.score}", True, (0,0,0))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)
        return True

    # --------------------------------------------------------------------------
    # Private Helper Methods for step()
    # --------------------------------------------------------------------------
    def _handle_player_movement(self, action):
        """Applies the chosen action to the player."""
        self.player.move(action)
        self.player.update()

    def _handle_collisions(self):
        """Checks for and handles collisions between the player and platforms."""
        landed = False
        if self.player.vel.y > 0:  # Player is moving down
            hits = pygame.sprite.spritecollide(self.player, self.platforms, False)
            if hits:
                # Land on the highest platform among the collided ones
                closest_hit = min(hits, key=lambda p: abs(self.player.rect.bottom - p.rect.top))
                # Ensure the player was above the platform in the previous frame to prevent phasing
                if self.player.rect.bottom < closest_hit.rect.top + self.player.vel.y + 1:
                    self.player.pos.y = closest_hit.rect.top
                    self.player.vel.y = 0
                    self.player.jump()
                    landed = True
        return landed

    def _scroll_camera(self):
        """Scrolls the camera if the player is moving up in the top quarter of the screen."""
        if self.player.rect.top <= SCREEN_HEIGHT / 4 and self.player.vel.y < 0:
            scroll_amount = -self.player.vel.y
            self.player.pos.y += scroll_amount
            self.camera_y += scroll_amount
            for plat in self.platforms:
                plat.rect.y += scroll_amount

    def _manage_platforms(self):
        """Removes old platforms and generates new ones."""
        # Remove platforms that have scrolled off the bottom of the screen
        for p in self.platforms.copy():
            if p.rect.top > SCREEN_HEIGHT:
                p.kill()

        # Generate new platforms if there are not enough
        if len(self.platforms) < 15:
            highest_plat = min(self.platforms, key=lambda p: p.rect.y, default=None)
            y_start = highest_plat.rect.y if highest_plat else self.player.pos.y
            new_platforms = self._generate_platforms(5, y_start)
            self.platforms.add(new_platforms)
            self.all_sprites.add(new_platforms)

    def _update_score(self):
        """Updates the score based on the maximum height reached."""
        height_score = int(self.camera_y / 10)
        self.score = max(self.score, height_score)

    def _check_game_over(self):
        """Checks if the game over condition is met."""
        if self.player.rect.top > SCREEN_HEIGHT:
            self.done = True

    def _calculate_reward(self, prev_score, landed):
        """Calculates the reward for the agent based on game events."""
        if self.done:
            return -10  # Large penalty for dying

        reward = 0
        # Reward for gaining height
        reward += (self.score - prev_score)
        # Small bonus for landing on a platform
        if landed:
            reward += 1
        return reward

    def _generate_platforms(self, num, y_start):
        """Generates a number of new platforms."""
        platforms = pygame.sprite.Group()
        for i in range(num):
            width = random.randint(PLATFORM_MIN_WIDTH, PLATFORM_MAX_WIDTH)
            x = random.randint(0, SCREEN_WIDTH - width)
            y = y_start - i * PLATFORM_SPACING
            platforms.add(Platform(x, y, width))
        return platforms

    ### NEW: Replaced the old _get_state method with the improved 6-D version.
    def _get_state(self):
        """
        Generates a state vector with information crucial for decision-making.
        State: [player_x_norm, player_x_vel_norm, player_y_vel_norm,
                target_dx, target_dy, target_width_norm] (6 values)
        """
        # --- Player Information ---
        # Normalize position to be -1 (left) to 1 (right)
        player_x_norm = (self.player.pos.x - SCREEN_WIDTH / 2) / (SCREEN_WIDTH / 2)
        # Normalize x velocity. We can use a heuristic max speed.
        player_x_vel_norm = np.clip(self.player.vel.x / (PLAYER_ACC * 5), -1, 1)
        # Normalize y velocity by the jump power
        player_y_vel_norm = np.clip(self.player.vel.y / abs(PLAYER_JUMP_POWER), -1, 1)

        state = [player_x_norm, player_x_vel_norm, player_y_vel_norm]

        # --- Target Platform Information ---
        # Find platforms that are potential future targets (above the player's head)
        platforms_above = [p for p in self.platforms if p.rect.bottom < self.player.rect.top]

        if platforms_above:
            # The target is the platform vertically closest to the player
            target_platform = min(platforms_above, key=lambda p: self.player.rect.top - p.rect.bottom)

            # Calculate relative distance to the center of the target platform
            target_dx = (target_platform.rect.centerx - self.player.pos.x) / SCREEN_WIDTH
            target_dy = (target_platform.rect.centery - self.player.pos.y) / SCREEN_HEIGHT

            # Normalize the platform's width (0 to 1)
            target_width_norm = (target_platform.rect.width - PLATFORM_MIN_WIDTH) / (PLATFORM_MAX_WIDTH - PLATFORM_MIN_WIDTH)

            state.extend([target_dx, target_dy, target_width_norm])
        else:
            # If no platforms are above, use default "far away" values
            state.extend([0, 1, 0]) # dx=0 (centered), dy=far, width=min

        return np.array(state)

# ==============================================================================
# 4. THE DDQN AGENT
# ==============================================================================

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DDQNAgent:
    ### CHANGE: The agent's __init__ is unchanged, but now it will receive its
    ### state_size and action_size from the environment dynamically.
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# ==============================================================================
# 5. THE TRAINING LOOP
# ==============================================================================

def train():
    os.makedirs("models", exist_ok=True)
    env = IcyTowerEnv()

    ### CHANGE: Dynamically get state and action sizes from the environment
    action_size = env.action_space_n
    # Get the state size by doing a reset and checking the length of the state vector
    initial_state = env.reset()
    state_size = len(initial_state)
    
    print(f"State size: {state_size}, Action size: {action_size}")

    agent = DDQNAgent(state_size=state_size, action_size=action_size, seed=0)

    n_episodes = 2000
    max_t = 10000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start
    scores = []
    scores_window = deque(maxlen=100)
    
    running = True
    for i_episode in range(1, n_episodes + 1):
        if not running:
            break

        state = env.reset()
        episode_reward = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            running = env.render()
            if not running:
                break
            
            if done:
                break
        
        scores_window.append(env.score)
        scores.append(env.score)
        eps = max(eps_end, eps_decay * eps)
        
        print(f'\rEpisode {i_episode}\tFinal Score: {env.score}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tFinal Score: {env.score}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}')

        if np.mean(scores_window) >= 500: # Adjust target score if needed
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'models/icy_tower_ddqn_improved.pth')
            break

    pygame.quit()
    return scores

# ==============================================================================
# 6. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    scores = train()

    print("\nTraining finished. Plotting scores...")
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    except ImportError:
        print("matplotlib not found, skipping plot.")