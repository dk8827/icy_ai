import random
import numpy as np
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_W, PLAYER_H, PLAYER_GRAV,
    PLAYER_ACC, PLAYER_FRICTION, PLAYER_JUMP, PLAT_MIN_W, PLAT_MAX_W,
    PLAT_H, PLAT_SPACING, COMBO_TIMEOUT_FRAMES
)

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

        if not landed and self.combo_floors and self.total_frames - self.last_land_frame > COMBO_TIMEOUT_FRAMES:
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