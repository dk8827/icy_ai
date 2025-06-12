# icy_tower_ddqn_exact.py
# A minimal Icy-Tower clone with authentic scoring + DDQN agent scaffold
# ----------------------------------------------------------------------
import pygame, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import random, math, os
import numpy as np
from collections import deque, namedtuple

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

COMBO_TIMEOUT_MS = 3000      # 3 s between landings to keep a combo alive

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Experience = namedtuple("Experience",
                        ("state", "action", "reward", "next_state", "done"))

# ========= 1. GAME COMPONENTS =========================================
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([PLAYER_W, PLAYER_H])
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.pos = pygame.math.Vector2(SCREEN_WIDTH/2, SCREEN_HEIGHT-50)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, PLAYER_GRAV)

    def move(self, action):                    # 0=L, 1=stay, 2=R
        self.acc.x = 0
        if action == 0: self.acc.x = -PLAYER_ACC
        if action == 2: self.acc.x =  PLAYER_ACC

    def update(self):
        self.acc.x += self.vel.x * PLAYER_FRICTION
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        self.pos.x %= SCREEN_WIDTH            # wrap-around
        self.rect.midbottom = self.pos

    def jump(self): self.vel.y = PLAYER_JUMP

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, floor_no):
        super().__init__()
        self.image = pygame.Surface([width, PLAT_H])
        self.image.fill(PLATFORM_COLOR)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.floor_no = floor_no               # crucial for scoring

# ========= 2. GAME ENVIRONMENT ========================================
class IcyTowerEnv:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Icy Tower DDQN")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont(None, 30)

        # dynamic action-space size (Gym-style)
        self.action_space_n = 3

    # ------------------------------------------------------------------
    def reset(self):
        self.player = Player()

        ground = Platform(0, SCREEN_HEIGHT-40, SCREEN_WIDTH, floor_no=0)
        self.platforms = pygame.sprite.Group(ground)
        self.all_sprites = pygame.sprite.Group(self.player, ground)

        self.next_floor_no = 1
        self.platforms.add(self._generate_platforms(15,
                         SCREEN_HEIGHT-100))
        self.all_sprites.add(*self.platforms)

        # scoring state -------------------------------------------------
        self.current_floor  = 0
        self.highest_floor  = 0
        self.combo_floors   = 0
        self.multi_jumps    = 0
        self.combo_points   = 0
        self.last_land_time = pygame.time.get_ticks()

        self.score = 0
        self.camera_y = 0
        self.done = False
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action):
        prev_score = self.score

        self._handle_player(action)
        landed = self._handle_collisions()
        self._scroll_camera()
        self._manage_platforms()

        # combo timeout check
        if not landed and self.combo_floors and \
           pygame.time.get_ticks() - self.last_land_time > COMBO_TIMEOUT_MS:
            self._finalize_combo()

        self._update_score()
        self._check_game_over()

        reward = self._calc_reward(prev_score, landed)
        return self._get_state(), reward, self.done, {}

    # ------------------------------------------------------------------
    def render(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.done = True
                return False
        self.screen.fill(BACKGROUND)
        self.all_sprites.draw(self.screen)
        txt = self.font.render(f"SCORE {self.score}", True, (0,0,0))
        self.screen.blit(txt, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)
        return True

    # ======= helpers ===================================================
    def _handle_player(self, action):
        self.player.move(action); self.player.update()

    def _handle_collisions(self):
        landed = False
        if self.player.vel.y > 0:   # falling
            hits = pygame.sprite.spritecollide(self.player, self.platforms, False)
            if hits:
                plat = min(hits,
                           key=lambda p: abs(self.player.rect.bottom - p.rect.top))
                if self.player.rect.bottom < plat.rect.top + self.player.vel.y + 1:
                    self.player.pos.y = plat.rect.top
                    self.player.vel.y = 0
                    self.player.jump()
                    self._on_land(plat)
                    landed = True
        return landed

    def _on_land(self, plat):
        diff = plat.floor_no - self.current_floor
        now  = pygame.time.get_ticks()

        if diff >= 2:                       # multi-floor jump
            self.combo_floors += diff
            self.multi_jumps  += 1
        else:                               # normal or downward landing
            self._finalize_combo()

        self.current_floor  = plat.floor_no
        self.highest_floor  = max(self.highest_floor, self.current_floor)
        self.last_land_time = now

    def _finalize_combo(self):
        if self.multi_jumps >= 2 and self.combo_floors >= 4:
            self.combo_points += self.combo_floors ** 2
        self.combo_floors = 0;  self.multi_jumps = 0

    def _scroll_camera(self):
        if self.player.rect.top <= SCREEN_HEIGHT/4 and self.player.vel.y < 0:
            dy = -self.player.vel.y
            self.player.pos.y += dy
            self.camera_y += dy
            for p in self.platforms: p.rect.y += dy

    def _manage_platforms(self):
        for p in self.platforms.copy():
            if p.rect.top > SCREEN_HEIGHT: p.kill()
        if len(self.platforms) < 15:
            top = min(self.platforms, key=lambda p: p.rect.y).rect.y
            new = self._generate_platforms(5, top)
            self.platforms.add(new); self.all_sprites.add(*new)

    def _update_score(self):
        self.score = 10*self.highest_floor + self.combo_points

    def _check_game_over(self):
        if self.player.rect.top > SCREEN_HEIGHT:
            self.done = True         # unfinished combo is void

    def _calc_reward(self, prev_score, landed):
        if self.done: return -10
        reward = self.score - prev_score
        if landed: reward += 1
        return reward

    def _generate_platforms(self, n, y_start):
        group = pygame.sprite.Group()
        for i in range(n):
            w = random.randint(PLAT_MIN_W, PLAT_MAX_W)
            x = random.randint(0, SCREEN_WIDTH-w)
            y = y_start - i*PLAT_SPACING
            group.add(Platform(x, y, w, self.next_floor_no))
            self.next_floor_no += 1
        return group

    # --------------- state vector (simple) ----------------------------
    def _get_state(self):
        px = (self.player.pos.x - SCREEN_WIDTH/2)/(SCREEN_WIDTH/2)
        vx = np.clip(self.player.vel.x/(PLAYER_ACC*5), -1, 1)
        vy = np.clip(self.player.vel.y/abs(PLAYER_JUMP), -1, 1)

        # nearest platform above
        above = [p for p in self.platforms if p.rect.bottom < self.player.rect.top]
        if above:
            tgt = min(above, key=lambda p: self.player.rect.top - p.rect.bottom)
            dx  = (tgt.rect.centerx - self.player.pos.x)/SCREEN_WIDTH
            dy  = (tgt.rect.centery - self.player.pos.y)/SCREEN_HEIGHT
            dw  = (tgt.rect.width - PLAT_MIN_W)/(PLAT_MAX_W - PLAT_MIN_W)
        else:
            dx, dy, dw = 0, 1, 0
        return np.array([px, vx, vy, dx, dy, dw], dtype=np.float32)

# ========= 3.  (Minimal) Agent & Training scaffold ====================
# You can plug your existing DDQN classes here; only environment changed.
# ----------------------------------------------------------------------
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
        # soft update
        for tp, lp in zip(self.tgt.parameters(), self.net.parameters()):
            tp.data.copy_(0.995*tp.data + 0.005*lp.data)

# ========= 4. TRAIN LOOP (very short demo) ============================
def train():
    env = IcyTowerEnv()
    s   = env.reset(); state_size = len(s)
    agent = DDQNAgent(state_size, env.action_space_n)

    eps = 1.0
    for ep in range(1, 201):
        s = env.reset(); rsum = 0
        while True:
            a = agent.act(s, eps)
            s2, r, d, _ = env.step(a)
            agent.step(s, a, r, s2, d)
            s = s2; rsum += r
            if not env.render(): return          # quit window
            if d: break
        eps = max(0.01, eps*0.995)
        print(f"Ep {ep:3d} | score {env.score:5d} | ep-reward {rsum:6.1f} | ε {eps:.3f}")

    pygame.quit()

# ========= 5. MAIN ====================================================
if __name__ == "__main__":
    train()
