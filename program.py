import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque
import numpy as np

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
STATE_SIZE = 7  # [player_x, player_y_vel, p1_dx, p1_dy, p2_dx, p2_dy, p3_dx]
ACTION_SIZE = 3 # 0: Left, 1: Stay, 2: Right

BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
LR = 5e-4               # Learning rate
TAU = 1e-3              # For soft update of target network
UPDATE_EVERY = 4        # How often to update the network
TARGET_UPDATE_FREQ = 100 # How often to update the target network

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# 1. THE GAME ENVIRONMENT (Simplified Icy Tower)
# ==============================================================================

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([PLAYER_WIDTH, PLAYER_HEIGHT])
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.pos = pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)

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

    def apply_gravity(self):
        self.vel.y += PLAYER_GRAVITY


class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width):
        super().__init__()
        self.image = pygame.Surface([width, PLATFORM_HEIGHT])
        self.image.fill(PLATFORM_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class IcyTowerEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(GAME_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

    def _generate_platforms(self, num, y_start):
        platforms = pygame.sprite.Group()
        for i in range(num):
            width = random.randint(PLATFORM_MIN_WIDTH, PLATFORM_MAX_WIDTH)
            x = random.randint(0, SCREEN_WIDTH - width)
            y = y_start - i * PLATFORM_SPACING
            platforms.add(Platform(x, y, width))
        return platforms
    
    def reset(self):
        self.player = Player()
        self.player.pos = pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50)

        # Initial platform
        initial_platform = Platform(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH)
        self.platforms = pygame.sprite.Group(initial_platform)
        
        # Procedurally generated platforms
        new_platforms = self._generate_platforms(15, SCREEN_HEIGHT - 100)
        self.platforms.add(new_platforms)
        
        self.all_sprites = pygame.sprite.Group(self.player, self.platforms)

        self.score = 0
        self.camera_y = 0
        self.done = False
        
        return self._get_state()

    def _get_state(self):
        # State: [player_x_norm, player_y_vel_norm, p1_dx, p1_dy, p2_dx, p2_dy, p3_dx, p3_dy]
        # Normalize player info
        player_x_norm = (self.player.pos.x - SCREEN_WIDTH / 2) / (SCREEN_WIDTH / 2)
        player_y_vel_norm = self.player.vel.y / 15.0 # Normalize by approx max jump velocity
        
        state = [player_x_norm, player_y_vel_norm]

        # Find 3 closest platforms above the player
        platforms_above = [p for p in self.platforms if p.rect.centery < self.player.rect.centery]
        platforms_above.sort(key=lambda p: self.player.pos.distance_to(p.rect.center))
        
        for i in range(2): # Look for the 2 closest platforms
            if i < len(platforms_above):
                p = platforms_above[i]
                dx = (p.rect.centerx - self.player.pos.x) / SCREEN_WIDTH
                dy = (p.rect.centery - self.player.pos.y) / SCREEN_HEIGHT
                state.extend([dx, dy])
            else:
                # If not enough platforms, use default far-away values
                state.extend([0, 1])

        # Add closest platform below the player (the one to land on)
        platforms_below = [p for p in self.platforms if p.rect.centery >= self.player.rect.centery]
        platforms_below.sort(key=lambda p: abs(p.rect.top - self.player.rect.bottom))
        if platforms_below:
            p = platforms_below[0]
            dx = (p.rect.centerx - self.player.pos.x) / SCREEN_WIDTH
            dy = (p.rect.top - self.player.rect.bottom) / SCREEN_HEIGHT
            state.append(dx)
        else:
            state.append(0)

        return np.array(state)


    def step(self, action):
        # --- Update Game Logic ---
        self.player.move(action)
        self.player.update()
        self.player.apply_gravity()

        # --- Collision Detection ---
        landed = False
        if self.player.vel.y > 0:
            hits = pygame.sprite.spritecollide(self.player, self.platforms, False)
            if hits:
                for hit in hits:
                    if self.player.rect.bottom <= hit.rect.bottom: # Land on top
                        self.player.pos.y = hit.rect.top
                        self.player.vel.y = 0
                        self.player.jump()
                        landed = True
                        break

        # --- Camera and Score Update ---
        prev_score = self.score
        if self.player.rect.top <= SCREEN_HEIGHT / 4:
            scroll = -self.player.vel.y
            self.player.pos.y -= scroll
            self.camera_y += scroll
            for plat in self.platforms:
                plat.rect.y -= scroll
                if plat.rect.top > SCREEN_HEIGHT:
                    plat.kill()
        
        self.score = max(self.score, int(-self.camera_y / 10))

        # --- Generate new platforms ---
        if len(self.platforms) < 15:
            last_plat = max(self.platforms, key=lambda p: p.rect.y, default=None)
            y_start = last_plat.rect.y if last_plat else self.player.pos.y
            
            new_platforms = self._generate_platforms(5, y_start)
            self.platforms.add(new_platforms)
            self.all_sprites.add(new_platforms)
        
        # --- Check for Game Over ---
        if self.player.rect.top > SCREEN_HEIGHT:
            self.done = True

        # --- Calculate Reward ---
        reward = 0
        if self.done:
            reward = -100 # Big penalty for dying
        else:
            reward = (self.score - prev_score) * 10 # Reward for climbing
            if landed:
                reward += 1 # Small reward for successful jump

        next_state = self._get_state()
        return next_state, reward, self.done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill(BACKGROUND_COLOR)
        self.all_sprites.draw(self.screen)
        
        # Draw Score
        score_text = self.font.render(f"Score: {self.score}", True, (0,0,0))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)

# ==============================================================================
# 2. THE DDQN AGENT
# ==============================================================================

class QNetwork(nn.Module):
    """Policy Model for mapping state -> action values."""
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
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
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

from collections import namedtuple

class DDQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict()) # Sync weights

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0 # for UPDATE_EVERY
        self.c_step = 0 # for TARGET_UPDATE_FREQ

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

        self.c_step = (self.c_step + 1) % TARGET_UPDATE_FREQ
        if self.c_step == 0:
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

        # Get best actions for next states from local model
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get Q values for those actions from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        
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

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# ==============================================================================
# 3. THE TRAINING LOOP
# ==============================================================================

def train():
    env = IcyTowerEnv()
    agent = DDQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0)

    n_episodes = 2000
    max_t = 10000 # Max timesteps per episode
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            # --- IMPORTANT: Render the game ---
            # You can comment this out for faster training
            env.render()
            
            if done:
                break
        
        scores_window.append(env.score)
        scores.append(env.score)
        eps = max(eps_end, eps_decay * eps)
        
        print(f'\rEpisode {i_episode}\tFinal Score: {env.score}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tFinal Score: {env.score}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}')

        # Save the model if it's performing well
        if np.mean(scores_window) >= 500: # Example goal
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'icy_tower_ddqn.pth')
            break

    pygame.quit()
    return scores

if __name__ == '__main__':
    scores = train()

    # Optional: Plotting scores after training
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