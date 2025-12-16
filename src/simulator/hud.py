import pygame
from ..common.constants import LSLChannel

class HUD:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont("Arial", 18)
        self.big_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.debug_mode = False
        
    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        
    def render(self, surface, all_probs, strategy_mgr, last_action):
        if self.debug_mode:
            self._render_debug(surface, all_probs, strategy_mgr, last_action)
        else:
            self._render_minimal(surface, all_probs, strategy_mgr, last_action)
            
    def _render_minimal(self, surface, all_probs, strategy_mgr, last_action):
        # Top Left: Status
        text = f"Stream: {strategy_mgr.selected_stream or 'None'} | Strategy: {strategy_mgr.get_active().name}"
        surf = self.big_font.render(text, True, (255, 255, 255))
        surface.blit(surf, (10, 10))
        
        # Action Indicator
        action_text = self._action_to_str(last_action)
        act_surf = self.big_font.render(f"Action: {action_text}", True, (0, 255, 0))
        surface.blit(act_surf, (10, 40))
        
        # F1 hint
        hint = self.font.render("Press F1 for Debug/Config", True, (150, 150, 150))
        surface.blit(hint, (10, self.height - 30))

    def _render_debug(self, surface, all_probs, strategy_mgr, last_action):
        # Dim background
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))
        
        # Title
        title = self.big_font.render("DEBUG HUD (F1)", True, (255, 255, 0))
        surface.blit(title, (10, 10))
        
        # Strategy Config
        strat = strategy_mgr.get_active()
        y = 50
        self._draw_text(surface, f"Strategy: {strat.name} (Press S to switch)", (10, y), color=(0, 255, 255))
        y += 25
        for k, v in strat.get_params().items():
            self._draw_text(surface, f"  {k}: {v:.2f} (Arrows +/-)", (10, y))
            y += 20
            
        y += 20
        self._draw_text(surface, "Streams (Press TAB to cycle driving stream):", (10, y), color=(0, 255, 255))
        y += 25
        
        classes = LSLChannel.names()
        
        if not all_probs:
            self._draw_text(surface, "  No streams detected...", (10, y), color=(255, 100, 100))
        
        for name, probs in all_probs.items():
            # Highlight selected
            is_selected = (name == strategy_mgr.selected_stream)
            color = (0, 255, 0) if is_selected else (200, 200, 200)
            prefix = ">> " if is_selected else "   "
            
            self._draw_text(surface, f"{prefix}{name}", (10, y), color=color)
            y += 20
            
            # Draw bars
            bar_w = 30
            bar_h = 80
            gap = 10
            start_x = 40
            
            for i, prob in enumerate(probs):
                # Class Name
                lbl = self.font.render(classes[i][:3], True, (200, 200, 200))
                surface.blit(lbl, (start_x + i*(bar_w+gap), y + bar_h + 2))
                
                # Bar
                h = int(prob * bar_h)
                rect = (start_x + i*(bar_w+gap), y + bar_h - h, bar_w, h)
                pygame.draw.rect(surface, (100, 100, 255), rect)
                pygame.draw.rect(surface, (255, 255, 255), rect, 1)
            
            y += bar_h + 25

    def _draw_text(self, surface, text, pos, color=(255, 255, 255)):
        surf = self.font.render(text, True, color)
        surface.blit(surf, pos)
        
    def _action_to_str(self, action):
        steer, gas, brake = action
        parts = []
        if steer < -0.1: parts.append("LEFT")
        elif steer > 0.1: parts.append("RIGHT")
        if gas > 0.1: parts.append("GAS")
        if brake > 0.1: parts.append("BRAKE")
        if not parts: return "RELAX"
        return "+".join(parts)
