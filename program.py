import os
import sys

try:
    import pygame
    from tqdm import tqdm
except ImportError:
    print("Pygame or tqdm not found. UI-based modes will be unavailable.")
    pygame = None

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, BACKGROUND, BUTTON_COLOR,
    BUTTON_HOVER, TEXT_COLOR, MODEL_PATH
)
from game_logic import IcyTowerLogic
from pygame_env import IcyTowerPygameEnv
from agent import DDQNAgent


def human_play():
    env = IcyTowerPygameEnv()
    env.reset()
    done = False

    running = True
    while running:
        action = 1  # Stay still by default
        if pygame is None: break
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  action = 0
        if keys[pygame.K_RIGHT]: action = 2

        if not done:
            _, _, done, _ = env.step(action)

        if not env.render():
            running = False

    env.close()


def train_agent(with_ui=True, num_episodes=200):
    env = IcyTowerPygameEnv() if with_ui else IcyTowerLogic()

    agent = DDQNAgent(env.state_size, env.action_space_n)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH} to continue training.")
        agent.load(MODEL_PATH)

    eps = 1.0
    ep_bar = tqdm(range(1, num_episodes + 1), desc="Episodes", unit="ep")
    for ep in ep_bar:
        s = env.reset()
        rsum = 0
        done = False

        last_score = 0
        steps_since_score_increase = 0
        max_steps_without_progress = 1000

        if not with_ui:
            step_bar = tqdm(desc=f"Ep {ep}", unit="step", leave=False)
        
        while not done:
            a = agent.act(s, eps)
            s2, r, done, _ = env.step(a)

            if not done:
                current_score = env.logic.score if with_ui else env.score
                if current_score > last_score:
                    last_score = current_score
                    steps_since_score_increase = 0
                else:
                    steps_since_score_increase += 1

                if steps_since_score_increase >= max_steps_without_progress:
                    done = True
                    # A brief message for the console, not cluttering the progress bar
                    tqdm.write(f"Episode {ep} stopped early: no score increase for {max_steps_without_progress} steps.")

            agent.step(s, a, r, s2, done)
            s = s2
            rsum += r

            if not with_ui:
                step_bar.update(1)
                score = env.score
                step_bar.set_postfix(score=f"{score}")

            if with_ui and not env.render():
                agent.save(MODEL_PATH)
                print(f"\nTraining stopped. Model saved to {MODEL_PATH}")
                env.close()
                return
        
        if not with_ui:
            step_bar.close()

        eps = max(0.01, eps * 0.995)
        score = env.logic.score if with_ui else env.score
        ep_bar.set_postfix(score=f"{score:5d}", reward=f"{rsum:6.1f}", eps=f"{eps:.3f}")

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
        a = agent.act(s, eps=0.0)
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
        "Learn with UI": (train_agent, True),
        "Learn without UI": (train_agent, False),
        "Play using AI": (ai_play, None),
    }
    button_rects = []
    y_start = SCREEN_HEIGHT / 2 - (len(buttons) * 60) / 2
    for i, (text, _) in enumerate(buttons.items()):
        rect = pygame.Rect(SCREEN_WIDTH / 2 - 150, y_start + i * 60, 300, 50)
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

                        pygame.quit()
                        if arg is not None:
                            func(with_ui=arg)
                        else:
                            func()

                        # Re-initialize for the menu
                        pygame.init()
                        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                        pygame.display.set_caption("Icy Tower - Main Menu")

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main_menu()