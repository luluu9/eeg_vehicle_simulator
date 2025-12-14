import gymnasium as gym
import pygame
from get_action import get_eeg_action


def get_action():

    keys = pygame.key.get_pressed()
    event_id = 1
    if keys[pygame.K_LEFT]:
        event_id = 2
    if keys[pygame.K_RIGHT]:
        event_id = 3
    if keys[pygame.K_UP]:
        event_id = 4
    if keys[pygame.K_DOWN]:
        event_id = 5

    eeg_action = get_eeg_action(event_id)
    return eeg_action

env = gym.make('CarRacing-v3', render_mode='human')

obs, info = env.reset()
done = False

while not done:
    action = get_action()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
