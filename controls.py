import pygame
import config
from core.utils import clamp


class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, v0, step=None, fmt="{:.2f}", font=None):
        self.rect = pygame.Rect(x, y, w, config.UI_SLIDER_HEIGHT_PX)
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.value = clamp(v0, vmin, vmax)
        self.step = step
        self.fmt = fmt
        self.dragging = False
        self.font = font

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
            self._set_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_mouse(event.pos[0])

    def _set_from_mouse(self, mx):
        x, y, w, h = self.rect
        t = clamp((mx - x) / w, 0.0, 1.0)
        val = self.vmin + t * (self.vmax - self.vmin)
        if self.step is not None:
            steps = round((val - self.vmin) / self.step)
            val = self.vmin + steps * self.step
        self.value = clamp(val, self.vmin, self.vmax)

    def draw(self, surf):
        x, y, w, h = self.rect
        pygame.draw.rect(
            surf,
            config.COLOR_SLIDER_TRACK,
            (x, y + h // 2 - config.UI_SLIDER_TRACK_HALF_THICKNESS_PX, w, config.UI_SLIDER_TRACK_HALF_THICKNESS_PX * 2),
            border_radius=config.UI_SLIDER_TRACK_HALF_THICKNESS_PX
        )
        t = (self.value - self.vmin) / (self.vmax - self.vmin)
        tx = x + int(t * w)
        pygame.draw.circle(surf, config.COLOR_SLIDER_THUMB_FILL, (tx, y + h // 2), config.UI_SLIDER_THUMB_RADIUS_PX)
        pygame.draw.circle(surf, config.COLOR_SLIDER_THUMB_BORDER, (tx, y + h // 2), config.UI_SLIDER_THUMB_RADIUS_PX, config.GRID_LINE_THIN_PX+1)
        
        if self.font:
            label_text = f"{self.label}: {self.fmt.format(self.value)}"
            surf.blit(self.font.render(label_text, True, config.COLOR_UI_TEXT), (x, y - config.UI_SLIDER_LABEL_OFFSET_Y_PX))


class Toggle:
    def __init__(self, x, y, label, value=False, font=None):
        self.rect = pygame.Rect(x, y, config.UI_TOGGLE_BOX_SIZE_PX, config.UI_TOGGLE_BOX_SIZE_PX)
        self.label = label
        self.value = value
        self.font = font

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and (self.rect.collidepoint(event.pos) or self.get_label_rect().collidepoint(event.pos)):
            self.value = not self.value

    def get_label_rect(self):
        if self.font:
            lbl = self.font.render(self.label, True, config.COLOR_UI_TEXT)
            r = lbl.get_rect()
            r.topleft = (self.rect.right + config.UI_TOGGLE_INSET_PX * 2, self.rect.top - config.GRID_LINE_THIN_PX)
            return r
        return pygame.Rect(0, 0, 0, 0)

    def draw(self, surf):
        pygame.draw.rect(surf, config.COLOR_TOGGLE_BG, self.rect, border_radius=config.GRID_LINE_THIN_PX+2)
        inner_rect = self.rect.inflate(-config.UI_TOGGLE_INSET_PX, -config.UI_TOGGLE_INSET_PX)
        pygame.draw.rect(surf, config.COLOR_TOGGLE_ON if self.value else config.COLOR_TOGGLE_OFF, inner_rect, border_radius=config.GRID_LINE_THIN_PX+2)
        
        if self.font:
            surf.blit(self.font.render(self.label, True, config.COLOR_UI_TEXT), (self.rect.right + config.UI_TOGGLE_INSET_PX * 2, self.rect.top - config.GRID_LINE_THIN_PX))


class Button:
    def __init__(self, x, y, w, h, label, font=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.clicked = False
        self.font = font

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.clicked = True

    def consume_click(self):
        was = self.clicked
        self.clicked = False
        return was

    def draw(self, surf):
        pygame.draw.rect(surf, config.COLOR_BUTTON_FILL, self.rect, border_radius=config.UI_BUTTON_CORNER_RADIUS_PX)
        pygame.draw.rect(surf, config.COLOR_BUTTON_BORDER, self.rect, config.UI_BUTTON_BORDER_THICKNESS_PX, border_radius=config.UI_BUTTON_CORNER_RADIUS_PX)
        
        if self.font:
            lbl = self.font.render(self.label, True, config.COLOR_UI_TEXT)
            surf.blit(lbl, (self.rect.centerx - lbl.get_width() // 2, self.rect.centery - lbl.get_height() // 2))
