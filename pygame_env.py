try:
    import pygame
except ImportError:
    pygame = None

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_W, PLAYER_H,
    PLAYER_COLOR, PLATFORM_COLOR, BACKGROUND, PLAT_H
)
from game_logic import IcyTowerLogic


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
    def __init__(self, screen=None):
        if pygame is None:
            raise ImportError("Pygame is required for UI-based modes.")

        self.was_screen_provided = screen is not None
        if self.was_screen_provided:
            self.screen = screen
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        pygame.display.set_caption("Icy Tower DDQN")
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
        return state, reward, done, info

    def _sync_sprites(self):
        """Re-create sprites from the logic state. Simple and robust."""
        self.platform_sprites.empty()
        self.all_sprites.empty()
        self.all_sprites.add(self.player_sprite)

        for p_logic in self.logic.platforms:
            p_sprite = PlatformSprite(p_logic)
            self.platform_sprites.add(p_sprite)
            self.all_sprites.add(p_sprite)

    def render(self):
        if pygame is None: return False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False

        self.player_sprite.rect.midbottom = self.logic.player.pos

        if len(self.platform_sprites) != len(self.logic.platforms):
             self._sync_sprites()

        for sprite in self.all_sprites:
            sprite.rect.y -= self.logic.camera_y

        self.screen.fill(BACKGROUND)
        self.all_sprites.draw(self.screen)
        txt = self.font.render(f"SCORE {self.logic.score}", True, (0,0,0))
        self.screen.blit(txt, (10, 10))

        if self.logic.hurry_up_flash_frames > 0:
            # Flashes by drawing only on some frames
            if (self.logic.total_frames // 10) % 2 == 0:
                hurry_font = pygame.font.SysFont(None, 50)
                hurry_text = hurry_font.render("Hurry Up!", True, (255, 0, 0))
                text_rect = hurry_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 4))
                self.screen.blit(hurry_text, text_rect)

        pygame.display.flip()

        for sprite in self.all_sprites:
            sprite.rect.y += self.logic.camera_y

        self.clock.tick(60)
        return True

    def close(self):
        if pygame and not self.was_screen_provided:
            pygame.quit() 