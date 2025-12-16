import gymnasium as gym
import pygame
import numpy as np
from .input_handler import MultiStreamMonitor
from .strategies import StrategyManager
from .hud import HUD

# Patch pygame to prevent calling .flip from inside step
def dummy_flip():
    pass

pygame_flip = pygame.display.flip
pygame.display.flip = dummy_flip


def main():
    # Initialize components
    monitor = MultiStreamMonitor()
    monitor.start()
    
    strategy_mgr = StrategyManager()
    
    # Setup environment
    env = gym.make("CarRacing-v3", render_mode="human", max_episode_steps=10_000_000)
    obs, info = env.reset()
    screen = pygame.display.get_surface()
    width, height = screen.get_size()
    hud = HUD(width, height)
    
    done = False
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    clock = pygame.time.Clock()
    
    print("Simulator started. Press F1 for Debug HUD.")
    
    while not done:
        # 1. Get Data
        all_probs = monitor.get_probabilities()
        
        # 2. Handle Inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    
                if event.key == pygame.K_F1:
                    hud.toggle_debug()
                    
                if hud.debug_mode:
                    # Config controls
                    if event.key == pygame.K_s:
                        strategy_mgr.next_strategy()
                        
                    if event.key == pygame.K_TAB:
                        # Cycle streams
                        streams = list(all_probs.keys())
                        if streams:
                            current = strategy_mgr.selected_stream
                            try:
                                idx = streams.index(current)
                                new_idx = (idx + 1) % len(streams)
                                strategy_mgr.selected_stream = streams[new_idx]
                            except ValueError:
                                strategy_mgr.selected_stream = streams[0]
                                
                    # Parameter tuning
                    strat = strategy_mgr.get_active()
                    # We tune "Threshold" by default if exists
                    # This is a bit hacky, ideally we select param
                    if "Threshold" in strat.get_params():
                        if event.key == pygame.K_UP:
                            strat.adjust_param("Threshold", 0.05)
                        if event.key == pygame.K_DOWN:
                            strat.adjust_param("Threshold", -0.05)
        
        # 3. Process Strategy
        # Note: If no streams, should returns relax (0,0,0)
        action = strategy_mgr.process(all_probs)
        print(action)
        # 4. Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 5. Render HUD
        hud.render(screen, all_probs, strategy_mgr, action)

        pygame_flip()
        clock.tick(60)

    monitor.stop()
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()