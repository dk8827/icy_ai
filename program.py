# icy_tower_ddqn_exact.py
# A minimal Icy-Tower clone with authentic scoring + DDQN agent scaffold
# REFACTORED to separate game logic from the Pygame UI
# ----------------------------------------------------------------------
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import random, math, os, sys
import numpy as np
from collections import deque, namedtuple

# Use a conditional import for pygame to ensure it's not a hard dependency
try:
    import pygame
except ImportError:
    print("Pygame not found. UI-based modes will be unavailable.")
    pygame = None # Allow headless mode to run

# ========= 0. CONFIG & CONSTANTS ======================================
SCREEN_WIDTH,  SCREEN_HEIGHT = 400, 600
PLAYER_W,      PLAYER_H      = 30, 40
PLAYER_ACC,    PLAYER_GRAV   = 0.8, 0.6
PLAYER_FRICTION, PLAYER_JUMP = -0.12, -15
PLAT_MIN_W,    PLAT_MAX_W    = 50, 120
PLAT_H,        PLAT_SPACING  = 20, 90

BACKGROUND     = (135, 206, 235)
PLAYER_COLOR   = (255,  0,   0)
PLATFORM_COLOR = (100, 100, 100)
BUTTON_COLOR   = (100, 100, 100)
BUTTON_HOVER   = (150, 150, 150)
TEXT_COLOR     = (255, 255, 255)

# Time is now measured in frames for the logic, not milliseconds
# 3000 ms @ 60 FPS = 180 frames
COMBO_TIMEOUT_FRAMES = 180

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/icy_tower_ddqn.pth"

Experience = namedtuple("Experience",
                        ("state", "action", "reward", "next_state", "done"))

# ========= 1. PURE GAME LOGIC (NO PYGAME!) ============================
# These are simple data classes, replacing pygame.sprite.Sprite
class LogicPlayer:
    def __init__(self):
        self.pos = [SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50]
        self.vel = [0, 0]
        self.acc = [0, PLAYER_GRAV]

    def get_rect(self): # Returns a simple (x, y, w, h) tuple
        return (self.pos[0] - PLAYER_W/2, self.pos[1] - PLAYER_H, PLAYER_W, PLAYER_H)

class LogicPlatform:
    def __init__(self, x, y, width, floor_no):
        self.x, self.y, self.width = x, y, width
        self.floor_no = floor_no

    def get_rect(self): # Returns a simple (x, y, w, h) tuple
        return (self.x, self.y, self.width, PLAT_H)

class IcyTowerLogic:
    """
    Manages the complete game state and rules without any graphics.
    This class is the "model" in a Model-View-Controller architecture.
    It can be run headlessly for fast training.
    """
    def __init__(self):
        self.action_space_n = 3
        self.state_size = 6 # px, vx, vy, dx, dy, dw

    def _rects_collide(self, rect1, rect2):
        # Simple Axis-Aligned Bounding Box collision check
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

    def reset(self):
        self.player = LogicPlayer()
        self.platforms = [LogicPlatform(0, SCREEN_HEIGHT-40, SCREEN_WIDTH, 0)]
        self.next_floor_no = 1
        self.platforms.extend(self._generate_platforms(15, SCREEN_HEIGHT-100))

        # Scoring & game state
        self.current_floor, self.highest_floor = 0, 0
        self.combo_floors, self.multi_jumps, self.combo_points = 0, 0, 0
        self.last_land_frame = 0
        self.total_frames = 0
        self.score, self.camera_y = 0, 0
        self.done = False
        return self._get_state()

    def step(self, action):
        prev_score = self.score
        self.total_frames += 1

        self._handle_player(action)
        landed = self._handle_collisions()
        self._scroll_camera()
        self._manage_platforms()

        if not landed and self.combo_floors and \
           self.total_frames - self.last_land_frame > COMBO_TIMEOUT_FRAMES:
            self._finalize_combo()

        self._update_score()
        self._check_game_over()
        reward = self._calc_reward(prev_score, landed)
        return self._get_state(), reward, self.done, {}

    def _handle_player(self, action):
        p = self.player
        p.acc[0] = 0
        if action == 0: p.acc[0] = -PLAYER_ACC  # Left
        if action == 2: p.acc[0] = PLAYER_ACC   # Right

        p.acc[0] += p.vel[0] * PLAYER_FRICTION
        p.vel[0] += p.acc[0]
        p.vel[1] += p.acc[1]
        p.pos[0] += p.vel[0] + 0.5 * p.acc[0]
        p.pos[1] += p.vel[1] + 0.5 * p.acc[1]
        p.pos[0] %= SCREEN_WIDTH # wrap-around

    def _handle_collisions(self):
        landed = False
        player_rect = self.player.get_rect()
        if self.player.vel[1] > 0: # Falling
            for plat in self.platforms:
                plat_rect = plat.get_rect()
                # A more precise collision check for landing
                if self._rects_collide(player_rect, plat_rect) and \
                   player_rect[1] + player_rect[3] < plat_rect[1] + self.player.vel[1] + 1:
                    self.player.pos[1] = plat_rect[1]
                    self.player.vel[1] = 0
                    self.player.vel[1] = PLAYER_JUMP # Jump
                    self._on_land(plat)
                    landed = True
                    break # Land on one platform at a time
        return landed

    def _on_land(self, plat):
        diff = plat.floor_no - self.current_floor
        if diff >= 2: # Multi-floor jump
            self.combo_floors += diff
            self.multi_jumps += 1
        else:
            self._finalize_combo()
        self.current_floor = plat.floor_no
        self.highest_floor = max(self.highest_floor, self.current_floor)
        self.last_land_frame = self.total_frames

    def _finalize_combo(self):
        if self.multi_jumps >= 2 and self.combo_floors >= 4:
            self.combo_points += self.combo_floors ** 2
        self.combo_floors, self.multi_jumps = 0, 0

    def _scroll_camera(self):
        player_y_on_screen = self.player.pos[1] - self.camera_y
        if player_y_on_screen < SCREEN_HEIGHT / 3:
            # Scroll up to keep the player in the bottom third of the screen
            self.camera_y = self.player.pos[1] - SCREEN_HEIGHT / 3


    def _manage_platforms(self):
        self.platforms = [p for p in self.platforms if p.y - self.camera_y < SCREEN_HEIGHT]
        if len(self.platforms) < 15:
            top_y = min(p.y for p in self.platforms)
            new_platforms = self._generate_platforms(5, top_y)
            self.platforms.extend(new_platforms)

    def _generate_platforms(self, n, y_start):
        plats = []
        for i in range(n):
            w = random.randint(PLAT_MIN_W, PLAT_MAX_W)
            x = random.randint(0, SCREEN_WIDTH - w)
            y = y_start - i * PLAT_SPACING - random.randint(0, 20) # Add some variance
            plats.append(LogicPlatform(x, y, w, self.next_floor_no))
            self.next_floor_no += 1
        return plats

    def _update_score(self):
        self.score = 10 * self.highest_floor + self.combo_points

    def _check_game_over(self):
        player_screen_y = self.player.pos[1] - self.camera_y
        if player_screen_y > SCREEN_HEIGHT:
            self.done = True

    def _calc_reward(self, prev_score, landed):
        if self.done: return -10
        reward = self.score - prev_score
        if landed: reward += 1
        return reward

    def _get_state(self):
        px = (self.player.pos[0] - SCREEN_WIDTH/2) / (SCREEN_WIDTH/2)
        vx = np.clip(self.player.vel[0] / (PLAYER_ACC*5), -1, 1)
        vy = np.clip(self.player.vel[1] / abs(PLAYER_JUMP), -1, 1)
        
        player_rect = self.player.get_rect()
        above = [p for p in self.platforms if p.get_rect()[1] + PLAT_H < player_rect[1]]
        if above:
            tgt = min(above, key=lambda p: player_rect[1] - (p.get_rect()[1] + PLAT_H))
            tgt_rect = tgt.get_rect()
            dx = (tgt_rect[0] + tgt_rect[2]/2 - self.player.pos[0]) / SCREEN_WIDTH
            dy = (tgt_rect[1] + tgt_rect[3]/2 - self.player.pos[1]) / SCREEN_HEIGHT
            dw = (tgt_rect[2] - PLAT_MIN_W) / (PLAT_MAX_W - PLAT_MIN_W)
        else:
            dx, dy, dw = 0, 1, 0
        return np.array([px, vx, vy, dx, dy, dw], dtype=np.float32)

    # Dummy methods to match the UI env interface for easy swapping
    def render(self): return True
    def close(self): pass


# ========= 2. PYGAME ENVIRONMENT (UI) =================================
# These are Pygame-specific sprite classes
class PlayerSprite(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([PLAYER_W, PLAYER_H])
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()

class PlatformSprite(pygame.sprite.Sprite):
    def __init__(self, platform_logic):
        super().__init__()
        self.logic_ref = platform_logic
        w, h = platform_logic.width, PLAT_H
        self.image = pygame.Surface([w, h])
        self.image.fill(PLATFORM_COLOR)
        self.rect = self.image.get_rect(topleft=(platform_logic.x, platform_logic.y))

class IcyTowerPygameEnv:
    """
    Manages the Pygame rendering and user input (the "view").
    It holds a reference to an IcyTowerLogic instance.
    """
    def __init__(self):
        if pygame is None:
            raise ImportError("Pygame is required for UI-based modes.")
        pygame.init()
        pygame.display.set_caption("Icy Tower DDQN")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)

        self.logic = IcyTowerLogic()
        self.action_space_n = self.logic.action_space_n
        self.state_size = self.logic.state_size

        self.player_sprite = PlayerSprite()
        self.platform_sprites = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group(self.player_sprite)

    def reset(self):
        state = self.logic.reset()
        self._sync_sprites()
        return state

    def step(self, action):
        state, reward, done, info = self.logic.step(action)
        if done:
            # Don't immediately close, allow one last render to show final state
            pass
        return state, reward, done, info

    def _sync_sprites(self):
        """Re-create sprites from the logic state. Simple and robust."""
        # This can be slow; a more optimized version would update existing sprites.
        # But for this game, it's perfectly fine and bug-resistant.
        self.platform_sprites.empty()
        self.all_sprites.empty()
        self.all_sprites.add(self.player_sprite)

        # Create new sprites for all current platforms in the logic
        for p_logic in self.logic.platforms:
            p_sprite = PlatformSprite(p_logic)
            self.platform_sprites.add(p_sprite)
            self.all_sprites.add(p_sprite)

    def render(self):
        # ==================================================================
        # THIS IS THE CORRECTED RENDER METHOD
        # ==================================================================
        if pygame is None: return False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
                return False

        # --- STAGE 1: Sync all Pygame sprites with the current logic state ---
        # Update player sprite's position to its absolute world coordinate.
        self.player_sprite.rect.midbottom = self.logic.player.pos

        # If the number of platforms has changed, recreate the platform sprites.
        # This is a simple way to ensure the view matches the model.
        if len(self.platform_sprites) != len(self.logic.platforms):
             self._sync_sprites()

        # --- STAGE 2: Apply the camera offset to all sprites for drawing ---
        # This temporarily moves their rects to the correct screen position.
        # It's done every frame and then reverted.
        for sprite in self.all_sprites:
            # We move the sprite DOWN by the camera's Y value.
            # This creates the illusion of the camera moving UP.
            sprite.rect.y -= self.logic.camera_y

        # --- STAGE 3: Draw the scene ---
        self.screen.fill(BACKGROUND)
        self.all_sprites.draw(self.screen)
        txt = self.font.render(f"SCORE {self.logic.score}", True, (0,0,0))
        self.screen.blit(txt, (10, 10))
        pygame.display.flip()

        # --- STAGE 4: Revert the camera offset ---
        # This resets the sprites' rects back to their absolute world coordinates
        # so they are correct for the next frame's calculations. THIS IS THE KEY FIX.
        for sprite in self.all_sprites:
            sprite.rect.y += self.logic.camera_y

        self.clock.tick(60)
        return True


    def close(self):
        if pygame:
            pygame.quit()


# ========= 3.  (Minimal) Agent & Training scaffold ====================
# This section is unchanged as it only interacts with the env interface
class QNetwork(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.fc1 = nn.Linear(s, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, a)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)

class ReplayBuffer:
    def __init__(self, size, batch):
        self.memory = deque(maxlen=size); self.batch = batch
    def add(self, *e): self.memory.append(Experience(*e))
    def sample(self):
        ex = random.sample(self.memory, self.batch)
        return Experience(*zip(*ex))
    def __len__(self): return len(self.memory)

class DDQNAgent:
    def __init__(self, s, a):
        self.nA = a
        self.net  = QNetwork(s, a).to(DEVICE)
        self.tgt  = QNetwork(s, a).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt  = optim.Adam(self.net.parameters(), lr=5e-4)
        self.mem  = ReplayBuffer(int(1e5), 64)
        self.t    = 0
    def act(self, state, eps):
        if random.random() < eps: return random.randrange(self.nA)
        with torch.no_grad():
            q = self.net(torch.tensor(state, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0))
        return int(torch.argmax(q).item())
    def step(self, s, a, r, s2, d):
        self.mem.add(s, a, r, s2, d)
        self.t = (self.t+1)%4
        if self.t==0 and len(self.mem)>=64: self.learn()
    def learn(self):
        batch = self.mem.sample()
        S  = torch.tensor(batch.state, dtype=torch.float32).to(DEVICE)
        A  = torch.tensor(batch.action).unsqueeze(1).to(DEVICE)
        R  = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        S2 = torch.tensor(batch.next_state, dtype=torch.float32).to(DEVICE)
        D  = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        next_a = self.net(S2).argmax(1, keepdim=True)
        Q_tgt  = self.tgt(S2).gather(1, next_a)
        Q_exp  = self.net(S).gather(1, A)
        Q_tar  = R + 0.99*Q_tgt*(1-D)
        loss   = F.mse_loss(Q_exp, Q_tar)

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        for tp, lp in zip(self.tgt.parameters(), self.net.parameters()):
            tp.data.copy_(0.995*tp.data + 0.005*lp.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.tgt.load_state_dict(self.net.state_dict())

# ========= 4. GAME MODES & MENU ========================================
def human_play():
    env = IcyTowerPygameEnv()
    s = env.reset()
    done = False
    
    # We need to manage the game loop slightly differently for human play
    # because we want to see the "Game Over" state.
    running = True
    while running:
        action = 1 # Stay still by default
        if pygame is None: break
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  action = 0
        if keys[pygame.K_RIGHT]: action = 2
        
        # Only step the game logic if it's not done
        if not done:
            s2, r, done, _ = env.step(action)
            s = s2
            
        # Always render, and check for quit events
        if not env.render(): 
            running = False # Exit if render returns False (e.g., window closed)
            
    env.close()


def train_agent(with_ui=True, num_episodes=200):
    # *** KEY CHANGE: Instantiate the correct environment ***
    env = IcyTowerPygameEnv() if with_ui else IcyTowerLogic()
    
    agent = DDQNAgent(env.state_size, env.action_space_n)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH} to continue training.")
        agent.load(MODEL_PATH)

    eps = 1.0
    for ep in range(1, num_episodes + 1):
        s = env.reset()
        rsum = 0
        done = False
        while not done:
            a = agent.act(s, eps)
            s2, r, done, _ = env.step(a)
            agent.step(s, a, r, s2, done)
            s = s2
            rsum += r
            if with_ui and not env.render(): # Render call also handles quit event
                agent.save(MODEL_PATH)
                print(f"\nTraining stopped. Model saved to {MODEL_PATH}")
                env.close()
                return

        eps = max(0.01, eps*0.995)
        # For headless, env.logic.score is the same as env.score
        score = env.logic.score if with_ui else env.score
        print(f"Ep {ep:3d} | score {score:5d} | ep-reward {rsum:6.1f} | ε {eps:.3f}", end='\r')
        if ep % 10 == 0 or ep == num_episodes: print() # Newline

    agent.save(MODEL_PATH)
    print(f"\nFinished training {num_episodes} episodes. Model saved to {MODEL_PATH}")
    env.close()

def ai_play():
    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}. Please train the AI first.")
        if pygame: pygame.time.wait(2000)
        return

    env = IcyTowerPygameEnv()
    agent = DDQNAgent(env.state_size, env.action_space_n)
    agent.load(MODEL_PATH)

    s = env.reset()
    done = False
    running = True
    while running:
        a = agent.act(s, eps=0.0) # Epsilon = 0 for deterministic play
        if not done:
            s2, r, done, _ = env.step(a)
            s = s2
        
        if not env.render(): 
            running = False
            
    env.close()

def draw_button(screen, rect, text, font, is_hovered):
    color = BUTTON_HOVER if is_hovered else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect)
    text_surf = font.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def main_menu():
    if pygame is None:
        print("Pygame not found. Cannot display main menu. Starting headless training.")
        try:
            train_agent(with_ui=False, num_episodes=1000)
        except KeyboardInterrupt:
            print("\nHeadless training interrupted by user.")
        sys.exit()
        
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Icy Tower - Main Menu")
    font = pygame.font.SysFont(None, 40)
    clock = pygame.time.Clock()

    buttons = {
        "Play Game (Keyboard)": (human_play, None),
        "Learn with UI":          (train_agent, True),
        "Learn without UI":       (train_agent, False),
        "Play using AI":          (ai_play, None),
    }
    button_rects = []
    y_start = SCREEN_HEIGHT / 2 - (len(buttons) * 60) / 2
    for i, (text, _) in enumerate(buttons.items()):
        rect = pygame.Rect(SCREEN_WIDTH/2 - 150, y_start + i * 60, 300, 50)
        button_rects.append(rect)

    running = True
    while running:
        screen.fill(BACKGROUND)
        mouse_pos = pygame.mouse.get_pos()
        
        for i, rect in enumerate(button_rects):
            is_hovered = rect.collidepoint(mouse_pos)
            draw_button(screen, rect, list(buttons.keys())[i], font, is_hovered)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(button_rects):
                    if rect.collidepoint(mouse_pos):
                        key = list(buttons.keys())[i]
                        func, arg = buttons[key]
                        
                        pygame.quit() # Quit the menu loop's pygame instance
                        if arg is not None:
                            func(with_ui=arg)
                        else:
                            func()
                        
                        # Re-initialize for the menu after the game mode finishes
                        pygame.init()
                        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                        pygame.display.set_caption("Icy Tower - Main Menu")

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

# ========= 5. MAIN ====================================================
if __name__ == "__main__":
    main_menu()