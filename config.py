# config.py
import torch
from collections import namedtuple

# Screen and Player dimensions
SCREEN_WIDTH,  SCREEN_HEIGHT = 400, 600
PLAYER_W,      PLAYER_H      = 30, 40
PLAT_MIN_W,    PLAT_MAX_W    = 50, 150
PLAT_H,        PLAT_SPACING  = 20, 90
PLAT_W_DECAY_RATE = 1
PLAT_W_RANDOMNESS = 15

# Physics constants
PLAYER_ACC,    PLAYER_GRAV   = 0.8, 0.6
PLAYER_FRICTION, PLAYER_JUMP = -0.12, -15

# Color constants
BACKGROUND     = (135, 206, 235)
PLAYER_COLOR   = (255,  0,   0)
PLATFORM_COLOR = (100, 100, 100)
BUTTON_COLOR   = (100, 100, 100)
BUTTON_HOVER   = (230, 230, 230)
TEXT_COLOR     = (0, 0, 0)

# Game logic constants
# 3000 ms @ 60 FPS = 180 frames
COMBO_TIMEOUT_FRAMES = 180
BASE_SCROLL_SPEED = 0.8
SCROLL_ACCELERATION = 0.25
SPEED_UP_INTERVAL_SECONDS = 30
MAX_SPEED_UPS = 5
HURRY_UP_FLASH_DURATION_FRAMES = 120  # 2 seconds at 60 FPS

# AI constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/icy_tower_model.pth"
NUM_EPISODES = 200

Experience = namedtuple("Experience",
                        ("state", "action", "reward", "next_state", "done")) 