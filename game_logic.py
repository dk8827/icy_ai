import random
import numpy as np
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_W, PLAYER_H, PLAYER_GRAV,
    PLAYER_ACC, PLAYER_FRICTION, PLAYER_JUMP, PLAT_MIN_W, PLAT_MAX_W,
    PLAT_H, PLAT_SPACING, COMBO_TIMEOUT_FRAMES, BASE_SCROLL_SPEED,
    SCROLL_ACCELERATION, SPEED_UP_INTERVAL_SECONDS, MAX_SPEED_UPS,
    HURRY_UP_FLASH_DURATION_FRAMES, PLAT_W_DECAY_RATE, PLAT_W_RANDOMNESS
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
        # The available actions are discrete
        # 0: left, 1: right, 2: stay
        # 3: left+jump, 4: right+jump, 5: stay+jump
        self.action_space_n = 6
        self.current_platform = None
        self.on_ground = True

        # To get the state size, we must first initialize the game to get a valid state.
        initial_state = self.reset()
        self.state_size = len(initial_state)

    def _rects_collide(self, rect1, rect2):
        # Simple Axis-Aligned Bounding Box collision check
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

    def reset(self):
        self.player = LogicPlayer()
        self.platforms = [LogicPlatform(0, SCREEN_HEIGHT-40, SCREEN_WIDTH, 0)]
        self.current_platform = self.platforms[0]
        self.next_floor_no = 1
        self.platforms.extend(self._generate_platforms(15, SCREEN_HEIGHT-100))

        # Scoring & game state
        self.current_floor, self.highest_floor = 0, 0
        self.combo_floors, self.multi_jumps, self.combo_points = 0, 0, 0
        self.last_land_frame = 0
        self.total_frames = 0
        self.score, self.camera_y = 0, 0
        self.done = False
        self.on_ground = True

        # New scrolling mechanics
        self.scrolling = False
        self.scroll_speed = 0
        self.last_speed_up_time = 0
        self.speed_ups = 0
        self.hurry_up_flash_frames = 0
        self.speed_up_interval_frames = SPEED_UP_INTERVAL_SECONDS * 60  # Assuming 60 FPS

        return self._get_state()

    def step(self, action):
        if self.hurry_up_flash_frames > 0:
            self.hurry_up_flash_frames -= 1

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
        player = self.player
        player.acc[0] = 0

        if self.on_ground:
            # Check if we have walked off the current platform
            if self.current_platform:
                plat_rect = self.current_platform.get_rect()
                player_rect = player.get_rect()
                player_left, _, player_w, _ = player_rect
                player_right = player_left + player_w
                plat_left, _, plat_w, _ = plat_rect
                plat_right = plat_left + plat_w
                if player_right < plat_left or player_left > plat_right:
                    self.on_ground = False

        # Horizontal Movement
        # Actions: 0:l, 1:r, 2:s, 3:l+j, 4:r+j, 5:s+j
        move_action = action % 3  # 0: left, 1: right, 2: stay
        if move_action == 0:
            player.acc[0] = -PLAYER_ACC  # Left
        elif move_action == 1:
            player.acc[0] = PLAYER_ACC   # Right

        # Jumping
        is_jump_action = action >= 3
        if is_jump_action and self.on_ground:
            player.vel[1] = PLAYER_JUMP
            self.on_ground = False

        player.acc[0] += player.vel[0] * PLAYER_FRICTION
        player.vel[0] += player.acc[0]
        player.vel[1] += player.acc[1]
        player.pos[0] += player.vel[0] + 0.5 * player.acc[0]
        player.pos[1] += player.vel[1] + 0.5 * player.acc[1]
        
        # Wall collisions
        if player.pos[0] < PLAYER_W / 2:
            player.pos[0] = PLAYER_W / 2
            player.vel[0] = 0
        elif player.pos[0] > SCREEN_WIDTH - PLAYER_W / 2:
            player.pos[0] = SCREEN_WIDTH - PLAYER_W / 2
            player.vel[0] = 0

    def _handle_collisions(self):
        landed = False
        player_rect = self.player.get_rect()
        if self.player.vel[1] >= 0: # Falling or at rest
            for plat in self.platforms:
                plat_rect = plat.get_rect()
                # A more precise collision check for landing
                if self._rects_collide(player_rect, plat_rect) and \
                   player_rect[1] + player_rect[3] < plat_rect[1] + self.player.vel[1] + 1:
                    self.player.pos[1] = plat_rect[1]
                    self.player.vel[1] = 0
                    self.on_ground = True
                    self._on_land(plat)
                    landed = True
                    break # Land on one platform at a time
        return landed

    def _on_land(self, plat):
        self.current_platform = plat
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
        # Determine if scrolling should start
        if not self.scrolling and self.highest_floor >= 5:
            self.scrolling = True
            self.scroll_speed = BASE_SCROLL_SPEED
            self.last_speed_up_time = self.total_frames

        # If scrolling is active, apply the base scroll and check for speed-ups
        if self.scrolling:
            if self.speed_ups < MAX_SPEED_UPS and \
               (self.total_frames - self.last_speed_up_time) > self.speed_up_interval_frames:
                self.scroll_speed += SCROLL_ACCELERATION
                self.speed_ups += 1
                self.last_speed_up_time = self.total_frames
                self.hurry_up_flash_frames = HURRY_UP_FLASH_DURATION_FRAMES
            
            # Move camera up by the scroll speed. (Subtract from y because y increases downwards)
            self.camera_y -= self.scroll_speed

        # Also, ensure the camera keeps the player in view if they are climbing fast
        player_y_on_screen = self.player.pos[1] - self.camera_y
        if player_y_on_screen < SCREEN_HEIGHT / 3:
            # This calculation ensures the camera moves higher (lower y value) to catch up with the player
            target_camera_y = self.player.pos[1] - SCREEN_HEIGHT / 3
            # Only adjust if it means moving the camera up further than the base scroll already has
            if target_camera_y < self.camera_y:
                self.camera_y = target_camera_y


    def _manage_platforms(self):
        self.platforms = [p for p in self.platforms if p.y - self.camera_y < SCREEN_HEIGHT]
        if len(self.platforms) < 15:
            top_y = min(p.y for p in self.platforms)
            new_platforms = self._generate_platforms(5, top_y)
            self.platforms.extend(new_platforms)

    def _generate_platforms(self, num_platforms, y_start):
        plats = []
        for i in range(num_platforms):
            base_width = PLAT_MAX_W - (self.next_floor_no * PLAT_W_DECAY_RATE)
            random_offset = random.uniform(-PLAT_W_RANDOMNESS, PLAT_W_RANDOMNESS)
            w = max(PLAT_MIN_W, base_width + random_offset)
            
            x_limit = SCREEN_WIDTH - int(w)
            x = random.randint(0, x_limit if x_limit > 0 else 0)

            y = y_start - i * PLAT_SPACING
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
        if landed: reward += 0.1
        return reward

    def _get_state(self):
        # Player state
        player_pos = self.player.pos
        player_vel = self.player.vel
        norm_player_x = (player_pos[0] - SCREEN_WIDTH/2) / (SCREEN_WIDTH/2)
        norm_player_vx = np.clip(player_vel[0] / (PLAYER_ACC*5), -1, 1)
        norm_player_vy = np.clip(player_vel[1] / abs(PLAYER_JUMP), -1, 1)
        player_state = [norm_player_x, norm_player_vx, norm_player_vy]

        # Current platform state
        current_platform_state = [0, 1, 0]  # Default to a far-away platform
        if self.current_platform:
            plat_rect = self.current_platform.get_rect()
            norm_dx = (plat_rect[0] + plat_rect[2]/2 - player_pos[0]) / SCREEN_WIDTH
            norm_dy = (plat_rect[1] - (player_pos[1] + PLAYER_H/2)) / SCREEN_HEIGHT
            norm_dw = (plat_rect[2] - PLAT_MIN_W) / (PLAT_MAX_W - PLAT_MIN_W)
            current_platform_state = [norm_dx, norm_dy, norm_dw]

        # Other platform states
        player_rect = self.player.get_rect()
        
        other_platforms = [p for p in self.platforms if p is not self.current_platform]

        # Separate platforms into those above and below the player's feet
        plats_below = sorted(
            [p for p in other_platforms if p.get_rect()[1] > player_rect[1] + player_rect[3]],
            key=lambda p: p.get_rect()[1]
        )
        plats_above = sorted(
            [p for p in other_platforms if p.get_rect()[1] + PLAT_H < player_rect[1]],
            key=lambda p: player_rect[1] - (p.get_rect()[1] + PLAT_H)
        )

        platform_states = []
        
        # 2 nearest platforms below
        for p in plats_below[:2]:
            p_rect = p.get_rect()
            norm_dx = (p_rect[0] + p_rect[2]/2 - player_pos[0]) / SCREEN_WIDTH
            norm_dy = (p_rect[1] - (player_pos[1] + PLAYER_H/2)) / SCREEN_HEIGHT
            norm_dw = (p_rect[2] - PLAT_MIN_W) / (PLAT_MAX_W - PLAT_MIN_W)
            platform_states.extend([norm_dx, norm_dy, norm_dw])
        # Pad if fewer than 2 platforms are below
        while len(platform_states) < 2 * 3:
            platform_states.extend([0, 1, 0]) # Represents a far-away platform

        # 3 nearest platforms above
        above_platform_features = []
        for p in plats_above[:3]:
            p_rect = p.get_rect()
            norm_dx = (p_rect[0] + p_rect[2]/2 - player_pos[0]) / SCREEN_WIDTH
            norm_dy = (p_rect[1] + PLAT_H/2 - player_pos[1]) / SCREEN_HEIGHT
            norm_dw = (p_rect[2] - PLAT_MIN_W) / (PLAT_MAX_W - PLAT_MIN_W)
            above_platform_features.extend([norm_dx, norm_dy, norm_dw])
        # Pad if fewer than 3 platforms are above
        while len(above_platform_features) < 3 * 3:
            above_platform_features.extend([0, 1, 0])

        platform_states.extend(above_platform_features)

        return np.array(player_state + current_platform_state + platform_states, dtype=np.float32)

    # Dummy methods to match the UI env interface for easy swapping
    def render(self): return True
    def close(self): pass 