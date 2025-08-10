import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pygame

DEG_FULL_CIRCLE: float = 360.0
DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi
MS_PER_SECOND: float = 1000.0
FRAME_RATE_TARGET_FPS: int = 60

WINDOW_WIDTH_PX: int = 1200
WINDOW_HEIGHT_PX: int = 800
CONTROL_PANEL_WIDTH_PX: int = 320
SCOPE_MARGIN_PX: int = 24
UI_PANEL_PADDING_PX: int = 24
UI_VERTICAL_SPACING_PX: int = 48
UI_TOGGLE_SPACING_PX: int = 26
UI_TOGGLE_EXTRA_GAP_PX: int = 32
UI_BUTTON_HEIGHT_PX: int = 28
UI_BUTTON_ROW_GAP_PX: int = 40
UI_BUTTON_COL_GAP_PX: int = 8
UI_RESET_GAP_PX: int = 36
UI_FOOTER_MARGIN_BOTTOM_PX: int = 28
UI_STATS_MARGIN_LEFT_PX: int = 24
UI_STATS_MARGIN_BOTTOM_PX: int = 28

FONT_SIZE_DEFAULT_PT: int = 16
FONT_SIZE_SMALL_PT: int = 12
FONT_SIZE_LARGE_PT: int = 18

RING_COUNT: int = 5
RING_LABEL_OFFSET_X_PX: int = 6
RING_LABEL_OFFSET_Y_PX: int = 8
GRID_LINE_THIN_PX: int = 1
SCOPE_FRAME_EXTRA_RADIUS_PX: int = 8
SCOPE_FRAME_THICKNESS_PX: int = 4
SCOPE_RING_THICKNESS_PX: int = 2
AZIMUTH_TICK_STEP_DEG: int = 15
AZIMUTH_TICK_INSET_PX: int = 6
SWEEP_POLY_SEGMENTS: int = 15
SWEEP_HEAD_THICKNESS_PX: int = 2
TRACK_TRAIL_MAXLEN: int = 40
TRACK_TRAIL_THICKNESS_PX: int = 2
VELOCITY_VECTOR_THICKNESS_PX: int = 2
TRACK_MARKER_OUTER_RADIUS_PX: int = 5
TRACK_MARKER_INNER_RADIUS_PX: int = 2
RAW_TARGET_MARKER_RADIUS_PX: int = 2
BLIP_CORE_RADIUS_PX: int = 3
BLIP_BLOOM_RADIUS_PX: int = 6
UI_SLIDER_HEIGHT_PX: int = 22
UI_SLIDER_THUMB_RADIUS_PX: int = 7
UI_SLIDER_TRACK_HALF_THICKNESS_PX: int = 2
UI_TOGGLE_BOX_SIZE_PX: int = 18
UI_TOGGLE_INSET_PX: int = 4
UI_BUTTON_CORNER_RADIUS_PX: int = 6
UI_BUTTON_BORDER_THICKNESS_PX: int = 2
UI_SLIDER_LABEL_OFFSET_Y_PX: int = 18

COLOR_BACKGROUND: Tuple[int, int, int] = (6, 12, 8)
COLOR_SCOPE_FRAME: Tuple[int, int, int] = (10, 40, 10)
COLOR_SCOPE_RING: Tuple[int, int, int] = (20, 70, 20)
COLOR_GRID_RING: Tuple[int, int, int] = (0, 60, 0)
COLOR_GRID_CROSS: Tuple[int, int, int] = (0, 60, 0)
COLOR_AZ_TICK: Tuple[int, int, int] = (0, 80, 0)
COLOR_RING_TEXT: Tuple[int, int, int] = (70, 180, 90)
COLOR_SWEEP_WEDGE_RGBA: Tuple[int, int, int, int] = (40, 200, 60, 32)
COLOR_SWEEP_HEAD_RGBA: Tuple[int, int, int, int] = (90, 255, 120, 200)
COLOR_BLIP_CORE_RGBA: Tuple[int, int, int, int] = (140, 255, 160, 230)
COLOR_BLIP_BLOOM_RGBA: Tuple[int, int, int, int] = (70, 200, 90, 150)
COLOR_RAW_TARGET_RGBA: Tuple[int, int, int, int] = (60, 110, 60, 120)
COLOR_TRACK_CONFIRMED: Tuple[int, int, int] = (80, 250, 120)
COLOR_TRACK_TENTATIVE: Tuple[int, int, int] = (140, 160, 90)
COLOR_TRAIL_RGBA: Tuple[int, int, int, int] = (70, 160, 110, 180)
COLOR_VELOCITY_RGBA: Tuple[int, int, int, int] = (120, 220, 150, 220)
COLOR_UI_PANEL_BG: Tuple[int, int, int] = (22, 22, 24)
COLOR_UI_PANEL_BORDER: Tuple[int, int, int] = (40, 40, 44)
COLOR_UI_TEXT: Tuple[int, int, int] = (230, 230, 230)
COLOR_UI_TEXT_MUTED: Tuple[int, int, int] = (180, 180, 180)
COLOR_UI_STATS_TEXT: Tuple[int, int, int] = (170, 210, 170)
COLOR_SLIDER_TRACK: Tuple[int, int, int] = (60, 60, 60)
COLOR_SLIDER_THUMB_FILL: Tuple[int, int, int] = (200, 200, 200)
COLOR_SLIDER_THUMB_BORDER: Tuple[int, int, int] = (20, 20, 20)
COLOR_TOGGLE_BG: Tuple[int, int, int] = (60, 60, 60)
COLOR_TOGGLE_ON: Tuple[int, int, int] = (90, 200, 90)
COLOR_TOGGLE_OFF: Tuple[int, int, int] = (30, 30, 30)
COLOR_BUTTON_FILL: Tuple[int, int, int] = (70, 70, 70)
COLOR_BUTTON_BORDER: Tuple[int, int, int] = (30, 30, 30)

UI_TITLE_TEXT: str = "Radar Controls"
UI_FOOTER_TEXT: str = "Esc: Quit   Space: Pause   . : Step"
UI_AZ_TRACKS_PREFIX: str = "Az: "

SPEED_RPM_TO_DEG_PER_SEC: float = 6.0 
MIN_AFTERGLOW_ALPHA: int = 0
MAX_AFTERGLOW_ALPHA: int = 255
AFTERGLOW_DEFAULT_ALPHA_DECAY_PER_FRAME: int = 14
AFTERGLOW_BLEND_MODE: int = pygame.BLEND_RGBA_SUB
SWEEP_VEL_SCALE_MIN: float = 0.4
SWEEP_VEL_SCALE_MAX: float = 1.2
TARGET_SPEED_SOFT_MAX_DIVISOR: float = 150.0
TARGET_MANEUVER_MIN_S: float = 2.0
TARGET_MANEUVER_MAX_S: float = 6.0
TARGET_ACCELERATION_MIN: float = -0.3
TARGET_ACCELERATION_MAX: float = 0.3

RADAR_DEFAULT_MAX_RANGE_M: float = 24000.0
RADAR_DEFAULT_RANGE_RES_M: float = 90.0
RADAR_DEFAULT_BEAMWIDTH_DEG: float = 3.0
RADAR_DEFAULT_RPM: float = 24.0
RADAR_DEFAULT_NOISE_POWER: float = 1.0
RADAR_DEFAULT_CLUTTER_ENABLED: bool = True
RADAR_DEFAULT_FALSE_ALARM_DENSITY: float = 0.0

CFAR_DEFAULT_GUARD_CELLS: int = 2
CFAR_DEFAULT_TRAINING_CELLS: int = 12
CFAR_DEFAULT_SCALE: float = 4.5
CFAR_PLATEAU_SUPPRESS_DISTANCE_BINS: int = 2

TRACKER_MEASUREMENT_STD_PX: float = 20.0
TRACKER_ASSOCIATION_GATE_PX: float = 60.0
TRACKER_CONFIRM_HITS_REQUIRED: int = 3
TRACKER_MAX_MISSES_ALLOWED: int = 20
TRACKER_INITIAL_POSITION_VAR: float = 100.0
TRACKER_INITIAL_VELOCITY_VAR: float = 100.0
TRACKER_PROCESS_NOISE_ACCEL: float = 2.0

TARGET_DEFAULT_COUNT: int = 14
TARGET_SPEED_MIN_FACTOR: float = 0.2
TARGET_SPEED_SPAWN_MIN_FACTOR: float = 0.3
TARGET_RANGE_SPAWN_MIN_M: float = 500.0
TARGET_RANGE_SPAWN_INNER_MIN_M: float = 1500.0
TARGET_RANGE_SPAWN_MAX_FACTOR: float = 0.95
TARGET_RANGE_SPAWN_BULK_MAX_FACTOR: float = 0.85
TARGET_RCS_DECADE_MIN: float = 2.0
TARGET_RCS_DECADE_MAX: float = 4.0

MEASUREMENT_RANGE_JITTER_STD_FACTOR: float = 0.4
MEASUREMENT_ANGLE_JITTER_STD_FACTOR: float = 0.25

SWEEP_WEDGE_ALPHA: int = COLOR_SWEEP_WEDGE_RGBA[3]
SWEEP_HEAD_ALPHA: int = COLOR_SWEEP_HEAD_RGBA[3]

LABEL_SHOW_RAW: str = "Show raw detections"
LABEL_CLUTTER: str = "Clutter on"
LABEL_TRAILS: str = "Show track trails"
LABEL_LABELS: str = "Show labels"
LABEL_SPAWN_BUTTON: str = "Spawn Target"
LABEL_REMOVE_BUTTON: str = "Remove Target"
LABEL_RESET_TRACKS: str = "Reset Tracks"
LABEL_PAUSE_STEP: str = "Pause / Step [Space]"

SLIDER_LABEL_MAX_RANGE_KM: str = "Max Range (km)"
SLIDER_LABEL_RPM: str = "Rotation (RPM)"
SLIDER_LABEL_BEAMWIDTH_DEG: str = "Beamwidth (deg)"
SLIDER_LABEL_RANGE_RES_M: str = "Range Res (m)"
SLIDER_LABEL_NOISE: str = "Noise Power"
SLIDER_LABEL_CFAR: str = "CFAR Scale"
SLIDER_LABEL_AFTERGLOW: str = "Afterglow"
SLIDER_LABEL_TARGETS: str = "Targets"
SLIDER_LABEL_MAX_SPEED: str = "Max Speed (m/s)"

SLIDER_MAX_RANGE_KM_MIN: float = 6.0
SLIDER_MAX_RANGE_KM_MAX: float = 60.0
SLIDER_MAX_RANGE_KM_STEP: float = 1.0

SLIDER_RPM_MIN: float = 6.0
SLIDER_RPM_MAX: float = 48.0
SLIDER_RPM_STEP: float = 1.0

SLIDER_BEAMWIDTH_MIN_DEG: float = 1.0
SLIDER_BEAMWIDTH_MAX_DEG: float = 10.0
SLIDER_BEAMWIDTH_STEP_DEG: float = 0.5

SLIDER_RANGE_RES_MIN_M: float = 15.0
SLIDER_RANGE_RES_MAX_M: float = 300.0
SLIDER_RANGE_RES_STEP_M: float = 5.0

SLIDER_NOISE_MIN: float = 0.2
SLIDER_NOISE_MAX: float = 4.0

SLIDER_CFAR_MIN: float = 2.0
SLIDER_CFAR_MAX: float = 10.0
SLIDER_CFAR_STEP: float = 0.1

SLIDER_AFTERGLOW_MIN: float = 4.0
SLIDER_AFTERGLOW_MAX: float = 40.0
SLIDER_AFTERGLOW_STEP: float = 1.0

SLIDER_TARGETS_MIN: int = 0
SLIDER_TARGETS_MAX: int = 40

SLIDER_MAX_SPEED_MIN_MPS: float = 30.0
SLIDER_MAX_SPEED_MAX_MPS: float = 250.0
SLIDER_MAX_SPEED_STEP_MPS: float = 5.0

def clamp(value: float, vmin: float, vmax: float) -> float:
    return vmax if value > vmax else vmin if value < vmin else value

def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + DEG_FULL_CIRCLE) % DEG_FULL_CIRCLE

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def pol2cart(radius: float, angle_deg: float) -> Tuple[float, float]:
    theta = angle_deg * DEG_TO_RAD
    return radius * math.cos(theta), radius * math.sin(theta)

def cart2pol(x: float, y: float) -> Tuple[float, float]:
    radius = math.hypot(x, y)
    angle_deg = math.atan2(y, x) * RAD_TO_DEG
    return radius, wrap_angle_deg(angle_deg)

pygame.init()
FONT = pygame.font.SysFont("consolas", FONT_SIZE_DEFAULT_PT)
FONT_SM = pygame.font.SysFont("consolas", FONT_SIZE_SMALL_PT)
FONT_LG = pygame.font.SysFont("consolas", FONT_SIZE_LARGE_PT, bold=True)

class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, v0, step=None, fmt="{:.2f}"):
        self.rect = pygame.Rect(x, y, w, UI_SLIDER_HEIGHT_PX)
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.value = clamp(v0, vmin, vmax)
        self.step = step
        self.fmt = fmt
        self.dragging = False

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
        # Track
        pygame.draw.rect(
            surf,
            COLOR_SLIDER_TRACK,
            (x, y + h // 2 - UI_SLIDER_TRACK_HALF_THICKNESS_PX, w, UI_SLIDER_TRACK_HALF_THICKNESS_PX * 2),
            border_radius=UI_SLIDER_TRACK_HALF_THICKNESS_PX
        )
        # Thumb
        t = (self.value - self.vmin) / (self.vmax - self.vmin)
        tx = x + int(t * w)
        pygame.draw.circle(surf, COLOR_SLIDER_THUMB_FILL, (tx, y + h // 2), UI_SLIDER_THUMB_RADIUS_PX)
        pygame.draw.circle(surf, COLOR_SLIDER_THUMB_BORDER, (tx, y + h // 2), UI_SLIDER_THUMB_RADIUS_PX, GRID_LINE_THIN_PX+1)

        # Text
        label_text = f"{self.label}: {self.fmt.format(self.value)}"
        surf.blit(FONT.render(label_text, True, COLOR_UI_TEXT), (x, y - UI_SLIDER_LABEL_OFFSET_Y_PX))

class Toggle:
    def __init__(self, x, y, label, value=False):
        self.rect = pygame.Rect(x, y, UI_TOGGLE_BOX_SIZE_PX, UI_TOGGLE_BOX_SIZE_PX)
        self.label = label
        self.value = value

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and (self.rect.collidepoint(event.pos) or self.get_label_rect().collidepoint(event.pos)):
            self.value = not self.value

    def get_label_rect(self):
        lbl = FONT.render(self.label, True, COLOR_UI_TEXT)
        r = lbl.get_rect()
        r.topleft = (self.rect.right + UI_TOGGLE_INSET_PX * 2, self.rect.top - GRID_LINE_THIN_PX)
        return r

    def draw(self, surf):
        pygame.draw.rect(surf, COLOR_TOGGLE_BG, self.rect, border_radius=GRID_LINE_THIN_PX+2)
        inner_rect = self.rect.inflate(-UI_TOGGLE_INSET_PX, -UI_TOGGLE_INSET_PX)
        pygame.draw.rect(surf, COLOR_TOGGLE_ON if self.value else COLOR_TOGGLE_OFF, inner_rect, border_radius=GRID_LINE_THIN_PX+2)
        surf.blit(FONT.render(self.label, True, COLOR_UI_TEXT), (self.rect.right + UI_TOGGLE_INSET_PX * 2, self.rect.top - GRID_LINE_THIN_PX))

class Button:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.clicked = False

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.clicked = True

    def consume_click(self):
        was = self.clicked
        self.clicked = False
        return was

    def draw(self, surf):
        pygame.draw.rect(surf, COLOR_BUTTON_FILL, self.rect, border_radius=UI_BUTTON_CORNER_RADIUS_PX)
        pygame.draw.rect(surf, COLOR_BUTTON_BORDER, self.rect, UI_BUTTON_BORDER_THICKNESS_PX, border_radius=UI_BUTTON_CORNER_RADIUS_PX)
        lbl = FONT.render(self.label, True, COLOR_UI_TEXT)
        surf.blit(lbl, (self.rect.centerx - lbl.get_width() // 2, self.rect.centery - lbl.get_height() // 2))

@dataclass
class Target:
    x: float
    y: float
    vx: float
    vy: float
    rcs: float
    maneuver_timer: float = 0.0

    def step(self, dt, world_radius_m):
        self.maneuver_timer -= dt
        if self.maneuver_timer <= 0.0:
            self.maneuver_timer = random.uniform(TARGET_MANEUVER_MIN_S, TARGET_MANEUVER_MAX_S)
            ax = random.uniform(TARGET_ACCELERATION_MIN, TARGET_ACCELERATION_MAX)
            ay = random.uniform(TARGET_ACCELERATION_MIN, TARGET_ACCELERATION_MAX)
            self.vx += ax
            self.vy += ay

        self.x += self.vx * dt
        self.y += self.vy * dt

        r, th = cart2pol(self.x, self.y)
        if r > world_radius_m:
            r = world_radius_m - (r - world_radius_m)
            self.x, self.y = pol2cart(r, th)

@dataclass
class Track:
    id: int
    x: np.ndarray   
    P: np.ndarray
    age: int = 0
    hits: int = 0
    misses: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=TRACK_TRAIL_MAXLEN))
    color: Tuple[int,int,int] = field(default_factory=lambda: (random.randint(80,255), random.randint(150,255), 100))
    confirmed: bool = False

class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.sigma_pos = TRACKER_MEASUREMENT_STD_PX
        self.gate_px = TRACKER_ASSOCIATION_GATE_PX
        self.confirm_hits = TRACKER_CONFIRM_HITS_REQUIRED
        self.max_misses = TRACKER_MAX_MISSES_ALLOWED
        self.q_accel = TRACKER_PROCESS_NOISE_ACCEL

    def predict(self, dt):
        F = np.array([[1.0, 0.0, dt,  0.0],
                      [0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=float)
        G = np.array([[0.5 * dt * dt, 0.0],
                      [0.0, 0.5 * dt * dt],
                      [dt, 0.0],
                      [0.0, dt]], dtype=float)
        Q = (self.q_accel ** 2) * (G @ G.T)

        for tr in self.tracks:
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + Q
            tr.age += 1
            tr.history.append((tr.x[0], tr.x[1]))

    def update(self, detections_xy: List[Tuple[float, float]]):
        unmatched = set(range(len(detections_xy)))
        for tr in self.tracks:
            tr.misses += 1

        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=float)
        R = np.diag([self.sigma_pos ** 2, self.sigma_pos ** 2])
        I = np.eye(4)

        for idx in list(unmatched):
            det = np.array(detections_xy[idx])
            best_track = None
            best_dist2 = self.gate_px ** 2
            for tr in self.tracks:
                pred_xy = tr.x[:2]
                d2 = np.sum((pred_xy - det) ** 2)
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_track = tr
            if best_track is not None:
                z = det
                y = z - H @ best_track.x
                S = H @ best_track.P @ H.T + R
                K = best_track.P @ H.T @ np.linalg.inv(S)
                best_track.x = best_track.x + K @ y
                best_track.P = (I - K @ H) @ best_track.P
                best_track.hits += 1
                best_track.misses = 0
                if not best_track.confirmed and best_track.hits >= self.confirm_hits:
                    best_track.confirmed = True
                unmatched.discard(idx)

        for idx in unmatched:
            det = detections_xy[idx]
            x0 = np.array([det[0], det[1], 0.0, 0.0], dtype=float)
            P0 = np.diag([
                TRACKER_INITIAL_POSITION_VAR,
                TRACKER_INITIAL_POSITION_VAR,
                TRACKER_INITIAL_VELOCITY_VAR,
                TRACKER_INITIAL_VELOCITY_VAR
            ])
            self.tracks.append(Track(self.next_id, x0, P0))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

class Radar:
    def __init__(self, scope_radius_px: int):
        self.scope_r_px = scope_radius_px
        self.max_range_m = RADAR_DEFAULT_MAX_RANGE_M
        self.range_res_m = RADAR_DEFAULT_RANGE_RES_M
        self.beamwidth_deg = RADAR_DEFAULT_BEAMWIDTH_DEG
        self.rpm = RADAR_DEFAULT_RPM
        self.angle_deg = 0.0

        self.noise_power = RADAR_DEFAULT_NOISE_POWER
        self.clutter_on = RADAR_DEFAULT_CLUTTER_ENABLED
        self.false_alarm_density = RADAR_DEFAULT_FALSE_ALARM_DENSITY

        self.cfar_guard = CFAR_DEFAULT_GUARD_CELLS
        self.cfar_train = CFAR_DEFAULT_TRAINING_CELLS
        self.cfar_scale = CFAR_DEFAULT_SCALE

        self.afterglow_decay = AFTERGLOW_DEFAULT_ALPHA_DECAY_PER_FRAME

        self.show_raw = False
        self.show_tracks = True
        self.show_labels = True
        self.show_trails = True

        self.max_off_boresight_for_return_deg = 6.0
        self.target_spread_bin_left_weight = 0.5
        self.target_spread_bin_center_weight = 1.0
        self.target_spread_bin_right_weight = 0.5
        self.radar_equation_scale = 3.2e17
        self.clutter_base_scale = 0.03
        self.clutter_sector_scale = 0.25
        self.false_alarm_power_min = 5.0
        self.false_alarm_power_max = 20.0

    def px_per_meter(self) -> float:
        return self.scope_r_px / self.max_range_m

    def meters_to_px(self, meters: float) -> float:
        return meters * self.px_per_meter()

    def step_angle(self, dt):
        self.angle_deg = wrap_angle_deg(self.angle_deg + self.rpm * SPEED_RPM_TO_DEG_PER_SEC * dt)

    def antenna_gain(self, offset_deg):
        fwhm_to_sigma = 2.355 
        sigma = self.beamwidth_deg / fwhm_to_sigma
        return math.exp(-0.5 * (offset_deg / sigma) ** 2)

    def simulate_beam_returns(self, angle_deg: float, targets: List[Target]) -> Tuple[np.ndarray, float]:
        Rmax = self.max_range_m
        dr = self.range_res_m
        nbins = int(Rmax // dr)
        returns = np.random.normal(0.0, self.noise_power, nbins)

        if self.clutter_on:
            r = np.arange(nbins) * dr + GRID_LINE_THIN_PX 
            clutter_profile = self.clutter_base_scale * (1.0 / np.sqrt(r))
            azimuthal_sector_mod = self.clutter_sector_scale * (1.0 + math.sin((angle_deg * 3.0) * DEG_TO_RAD))
            returns += clutter_profile * azimuthal_sector_mod * np.random.randn(nbins)

        for tg in targets:
            rng_m, bearing_deg = cart2pol(tg.x, tg.y)
            if rng_m > Rmax:
                continue
            off = min(abs(wrap_angle_deg(bearing_deg - angle_deg)),
                      DEG_FULL_CIRCLE - abs(wrap_angle_deg(bearing_deg - angle_deg)))
            if off > max(self.beamwidth_deg * 2.0, self.max_off_boresight_for_return_deg):
                continue
            gain = self.antenna_gain(off)
            snr_like = (tg.rcs * gain) / max(rng_m ** 4, 1.0) * self.radar_equation_scale
            bi = int(rng_m // dr)
            if (bi - 1) >= 0 and (bi + 1) < nbins:
                returns[bi - 1] += self.target_spread_bin_left_weight * snr_like
                returns[bi]     += self.target_spread_bin_center_weight * snr_like
                returns[bi + 1] += self.target_spread_bin_right_weight * snr_like

        if self.false_alarm_density > 0.0:
            nfa = np.random.poisson(self.false_alarm_density)
            for _ in range(nfa):
                idx = np.random.randint(0, nbins)
                returns[idx] += np.random.uniform(self.false_alarm_power_min, self.false_alarm_power_max) * self.noise_power

        return returns, dr

    def cfar_detect(self, arr: np.ndarray) -> List[int]:
        G, T = self.cfar_guard, self.cfar_train
        N = len(arr)
        hits = []
        for i in range(T + G, N - (T + G)):
            leading = arr[i - (T + G): i - G]
            trailing = arr[i + G + 1: i + G + 1 + T]
            noise_est = (np.mean(np.abs(leading)) + np.mean(np.abs(trailing))) * 0.5
            thr = self.cfar_scale * noise_est
            if arr[i] > thr:
                hits.append(i)

        pruned = []
        last_idx = - (CFAR_PLATEAU_SUPPRESS_DISTANCE_BINS + 10) 
        for idx in hits:
            if idx - last_idx > CFAR_PLATEAU_SUPPRESS_DISTANCE_BINS:
                pruned.append(idx)
            last_idx = idx
        return pruned

class App:
    def __init__(self):
        self.W, self.H = WINDOW_WIDTH_PX, WINDOW_HEIGHT_PX
        self.panel_w = CONTROL_PANEL_WIDTH_PX
        self.cx, self.cy = (self.W - self.panel_w) // 2, self.H // 2
        self.scope_r = min(self.cx, self.cy) - SCOPE_MARGIN_PX

        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("High-Fidelity Interactive Radar Simulation (No Magic Numbers)")
        self.clock = pygame.time.Clock()

        self.ppi = pygame.Surface((self.W - self.panel_w, self.H), pygame.SRCALPHA)
        self.overlay = pygame.Surface((self.W - self.panel_w, self.H), pygame.SRCALPHA)

        self.radar = Radar(self.scope_r)
        self.tracker = Tracker()

        self.targets: List[Target] = []
        self.target_count = TARGET_DEFAULT_COUNT
        self.target_speed_max = SLIDER_MAX_SPEED_MAX_MPS - (SLIDER_MAX_SPEED_MAX_MPS - SLIDER_MAX_SPEED_MIN_MPS) / 2.0
        self.spawn_targets(self.target_count)

        x0 = self.W - self.panel_w + UI_PANEL_PADDING_PX
        y = UI_PANEL_PADDING_PX
        w = self.panel_w - UI_PANEL_PADDING_PX * 2

        self.s_max_range = Slider(x0, y, w, SLIDER_LABEL_MAX_RANGE_KM,
                                  SLIDER_MAX_RANGE_KM_MIN, SLIDER_MAX_RANGE_KM_MAX,
                                  self.radar.max_range_m / 1000.0, step=SLIDER_MAX_RANGE_KM_STEP, fmt="{:.0f}"); y += UI_VERTICAL_SPACING_PX
        self.s_rpm       = Slider(x0, y, w, SLIDER_LABEL_RPM,
                                  SLIDER_RPM_MIN, SLIDER_RPM_MAX,
                                  self.radar.rpm, step=SLIDER_RPM_STEP, fmt="{:.0f}"); y += UI_VERTICAL_SPACING_PX
        self.s_bw        = Slider(x0, y, w, SLIDER_LABEL_BEAMWIDTH_DEG,
                                  SLIDER_BEAMWIDTH_MIN_DEG, SLIDER_BEAMWIDTH_MAX_DEG,
                                  self.radar.beamwidth_deg, step=SLIDER_BEAMWIDTH_STEP_DEG, fmt="{:.1f}"); y += UI_VERTICAL_SPACING_PX
        self.s_res       = Slider(x0, y, w, SLIDER_LABEL_RANGE_RES_M,
                                  SLIDER_RANGE_RES_MIN_M, SLIDER_RANGE_RES_MAX_M,
                                  self.radar.range_res_m, step=SLIDER_RANGE_RES_STEP_M, fmt="{:.0f}"); y += UI_VERTICAL_SPACING_PX
        self.s_noise     = Slider(x0, y, w, SLIDER_LABEL_NOISE,
                                  SLIDER_NOISE_MIN, SLIDER_NOISE_MAX,
                                  self.radar.noise_power, fmt="{:.2f}"); y += UI_VERTICAL_SPACING_PX
        self.s_cfar      = Slider(x0, y, w, SLIDER_LABEL_CFAR,
                                  SLIDER_CFAR_MIN, SLIDER_CFAR_MAX,
                                  self.radar.cfar_scale, step=SLIDER_CFAR_STEP, fmt="{:.1f}"); y += UI_VERTICAL_SPACING_PX
        self.s_after     = Slider(x0, y, w, SLIDER_LABEL_AFTERGLOW,
                                  SLIDER_AFTERGLOW_MIN, SLIDER_AFTERGLOW_MAX,
                                  self.radar.afterglow_decay, step=SLIDER_AFTERGLOW_STEP, fmt="{:.0f}"); y += UI_VERTICAL_SPACING_PX
        self.s_tgt_count = Slider(x0, y, w, SLIDER_LABEL_TARGETS,
                                  SLIDER_TARGETS_MIN, SLIDER_TARGETS_MAX,
                                  self.target_count, step=1.0, fmt="{:.0f}"); y += UI_VERTICAL_SPACING_PX
        self.s_tgt_spd   = Slider(x0, y, w, SLIDER_LABEL_MAX_SPEED,
                                  SLIDER_MAX_SPEED_MIN_MPS, SLIDER_MAX_SPEED_MAX_MPS,
                                  self.target_speed_max, step=SLIDER_MAX_SPEED_STEP_MPS, fmt="{:.0f}"); y += UI_VERTICAL_SPACING_PX

        y += GRID_LINE_THIN_PX * 8
        self.t_show_raw   = Toggle(x0, y, LABEL_SHOW_RAW, False); y += UI_TOGGLE_SPACING_PX
        self.t_clutter    = Toggle(x0, y, LABEL_CLUTTER, True); y += UI_TOGGLE_SPACING_PX
        self.t_trails     = Toggle(x0, y, LABEL_TRAILS, True); y += UI_TOGGLE_SPACING_PX
        self.t_labels     = Toggle(x0, y, LABEL_LABELS, True); y += UI_TOGGLE_EXTRA_GAP_PX

        half_w = (w - UI_BUTTON_COL_GAP_PX) // 2
        self.b_add_tgt    = Button(x0, y, half_w, UI_BUTTON_HEIGHT_PX, LABEL_SPAWN_BUTTON)
        self.b_rem_tgt    = Button(x0 + half_w + UI_BUTTON_COL_GAP_PX, y, half_w, UI_BUTTON_HEIGHT_PX, LABEL_REMOVE_BUTTON); y += UI_BUTTON_ROW_GAP_PX
        self.b_reset_tr   = Button(x0, y, w, UI_BUTTON_HEIGHT_PX, LABEL_RESET_TRACKS); y += UI_RESET_GAP_PX
        self.b_pause      = Button(x0, y, w, UI_BUTTON_HEIGHT_PX, LABEL_PAUSE_STEP); y += UI_RESET_GAP_PX

        self.ui_elements = [
            self.s_max_range, self.s_rpm, self.s_bw, self.s_res, self.s_noise,
            self.s_cfar, self.s_after, self.s_tgt_count, self.s_tgt_spd,
            self.t_show_raw, self.t_clutter, self.t_trails, self.t_labels,
            self.b_add_tgt, self.b_rem_tgt, self.b_reset_tr, self.b_pause
        ]

        self.paused = False
        self.step_once = False

    def spawn_targets(self, n):
        self.targets.clear()
        for _ in range(n):
            r = random.uniform(TARGET_RANGE_SPAWN_MIN_M, self.radar.max_range_m * TARGET_RANGE_SPAWN_BULK_MAX_FACTOR)
            th = random.uniform(0.0, DEG_FULL_CIRCLE)
            x, y = pol2cart(r, th)
            speed = random.uniform(self.target_speed_max * TARGET_SPEED_MIN_FACTOR, self.target_speed_max)
            heading = random.uniform(0.0, DEG_FULL_CIRCLE)
            vx, vy = pol2cart(speed, heading)
            rcs = 10 ** random.uniform(TARGET_RCS_DECADE_MIN, TARGET_RCS_DECADE_MAX)
            self.targets.append(Target(x, y, vx, vy, rcs))

    def add_one_target(self):
        r = random.uniform(TARGET_RANGE_SPAWN_INNER_MIN_M, self.radar.max_range_m * TARGET_RANGE_SPAWN_MAX_FACTOR)
        th = random.uniform(0.0, DEG_FULL_CIRCLE)
        x, y = pol2cart(r, th)
        speed = random.uniform(self.target_speed_max * TARGET_SPEED_SPAWN_MIN_FACTOR, self.target_speed_max)
        heading = random.uniform(0.0, DEG_FULL_CIRCLE)
        vx, vy = pol2cart(speed, heading)
        rcs = 10 ** random.uniform(TARGET_RCS_DECADE_MIN, TARGET_RCS_DECADE_MAX)
        self.targets.append(Target(x, y, vx, vy, rcs))

    def remove_one_target(self):
        if self.targets:
            self.targets.pop(random.randrange(len(self.targets)))

    def draw_scope_grid(self, surf):
        cx, cy, R = self.cx, self.cy, self.scope_r
        for i in range(1, RING_COUNT + 1):
            r = int(R * i / RING_COUNT)
            pygame.draw.circle(surf, COLOR_GRID_RING, (cx, cy), r, GRID_LINE_THIN_PX)
            rng_km = self.radar.max_range_m * (i / RING_COUNT) / 1000.0
            lbl = FONT_SM.render(f"{rng_km:.0f} km", True, COLOR_RING_TEXT)
            surf.blit(lbl, (cx + r + RING_LABEL_OFFSET_X_PX, cy - RING_LABEL_OFFSET_Y_PX))
        pygame.draw.line(surf, COLOR_GRID_CROSS, (cx - R, cy), (cx + R, cy), GRID_LINE_THIN_PX)
        pygame.draw.line(surf, COLOR_GRID_CROSS, (cx, cy - R), (cx, cy + R), GRID_LINE_THIN_PX)
        for deg in range(0, int(DEG_FULL_CIRCLE), AZIMUTH_TICK_STEP_DEG):
            x1, y1 = pol2cart(R - AZIMUTH_TICK_INSET_PX, deg)
            x2, y2 = pol2cart(R, deg)
            pygame.draw.line(surf, COLOR_AZ_TICK, (cx + int(x1), cy + int(y1)), (cx + int(x2), cy + int(y2)), GRID_LINE_THIN_PX)

    def fade_afterglow(self):
        decay = int(self.radar.afterglow_decay)
        decay = clamp(decay, MIN_AFTERGLOW_ALPHA, MAX_AFTERGLOW_ALPHA)
        if decay <= 0:
            return
        fade = pygame.Surface(self.ppi.get_size(), pygame.SRCALPHA)
        fade.fill((0, 0, 0, int(decay)))
        self.ppi.blit(fade, (0, 0), special_flags=AFTERGLOW_BLEND_MODE)

    def draw_sweep(self, angle_deg):
        bw = self.radar.beamwidth_deg
        R = self.scope_r
        cx, cy = self.cx, self.cy
        left = wrap_angle_deg(angle_deg - bw / 2.0)
        right = wrap_angle_deg(angle_deg + bw / 2.0)
        pts = [(cx, cy)]
        for t in np.linspace(left, right, SWEEP_POLY_SEGMENTS):
            x, y = pol2cart(R, t)
            pts.append((cx + int(x), cy + int(y)))
        pygame.draw.polygon(self.overlay, COLOR_SWEEP_WEDGE_RGBA, pts)
        hx, hy = pol2cart(R, angle_deg)
        pygame.draw.line(self.overlay, COLOR_SWEEP_HEAD_RGBA, (cx, cy), (cx + int(hx), cy + int(hy)), SWEEP_HEAD_THICKNESS_PX)

    def draw_blip(self, x_px, y_px, strong=True):
        core_alpha = COLOR_BLIP_CORE_RGBA[3] if strong else int(COLOR_BLIP_CORE_RGBA[3] * 0.52)
        core_color = (COLOR_BLIP_CORE_RGBA[0], COLOR_BLIP_CORE_RGBA[1], COLOR_BLIP_CORE_RGBA[2], core_alpha)
        pygame.draw.circle(self.ppi, core_color, (x_px, y_px), BLIP_CORE_RADIUS_PX)
        pygame.draw.circle(self.ppi, COLOR_BLIP_BLOOM_RGBA, (x_px, y_px), BLIP_BLOOM_RADIUS_PX, GRID_LINE_THIN_PX+1)

    def world_to_px(self, x_m, y_m):
        scale = self.radar.px_per_meter()
        return int(self.cx + x_m * scale), int(self.cy + y_m * scale)

    def run(self):
        while True:
            dt = self.clock.tick(FRAME_RATE_TARGET_FPS) / MS_PER_SECOND
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if ev.key == pygame.K_PERIOD:
                        self.step_once = True
                for el in self.ui_elements:
                    if hasattr(el, "handle"): el.handle(ev)

            # Apply UI parameter changes
            self.radar.max_range_m = self.s_max_range.value * 1000.0
            self.radar.rpm = self.s_rpm.value
            self.radar.beamwidth_deg = self.s_bw.value
            self.radar.range_res_m = self.s_res.value
            self.radar.noise_power = self.s_noise.value
            self.radar.cfar_scale = self.s_cfar.value
            self.radar.afterglow_decay = int(self.s_after.value)

            self.radar.show_raw = self.t_show_raw.value
            self.radar.clutter_on = self.t_clutter.value
            self.radar.show_trails = self.t_trails.value
            self.radar.show_labels = self.t_labels.value

            new_tgt_count = int(self.s_tgt_count.value)
            if new_tgt_count != len(self.targets):
                if new_tgt_count > len(self.targets):
                    for _ in range(new_tgt_count - len(self.targets)):
                        self.add_one_target()
                else:
                    for _ in range(len(self.targets) - new_tgt_count):
                        self.remove_one_target()
            self.target_speed_max = self.s_tgt_spd.value

            if self.b_add_tgt.consume_click(): self.add_one_target()
            if self.b_rem_tgt.consume_click(): self.remove_one_target()
            if self.b_reset_tr.consume_click(): self.tracker = Tracker()
            if self.b_pause.consume_click(): self.paused = not self.paused

            do_step = (not self.paused) or self.step_once
            if do_step:
                self.step_once = False
                self.radar.step_angle(dt)
                world_r = self.radar.max_range_m

                for t in self.targets:
                    speed = math.hypot(t.vx, t.vy)
                    if speed > self.target_speed_max:
                        scale = self.target_speed_max / (speed + 1e-6)
                        t.vx *= scale
                        t.vy *= scale
                    t.step(dt, world_r)

                returns, dr = self.radar.simulate_beam_returns(self.radar.angle_deg, self.targets)
                hits = self.radar.cfar_detect(returns)

                det_xy = []
                for bi in hits:
                    rng = (bi + 0.5) * dr
                    if rng > self.radar.max_range_m:
                        continue
                    rng += np.random.normal(0.0, self.radar.range_res_m * MEASUREMENT_RANGE_JITTER_STD_FACTOR)
                    ang = self.radar.angle_deg + np.random.normal(0.0, self.radar.beamwidth_deg * MEASUREMENT_ANGLE_JITTER_STD_FACTOR)
                    x_m, y_m = pol2cart(rng, ang)
                    x_px, y_px = self.world_to_px(x_m, y_m)
                    if math.hypot(x_px - self.cx, y_px - self.cy) <= self.scope_r:
                        self.draw_blip(x_px, y_px, strong=True)
                        det_xy.append(np.array([x_px, y_px], dtype=float))

                self.tracker.predict(dt)
                self.tracker.update(det_xy)
                self.fade_afterglow()

            # Compose frame
            self.overlay.fill((0, 0, 0, 0))
            self.draw_sweep(self.radar.angle_deg)

            if self.radar.show_raw:
                for t in self.targets:
                    x_px, y_px = self.world_to_px(t.x, t.y)
                    if math.hypot(x_px - self.cx, y_px - self.cy) <= self.scope_r:
                        pygame.draw.circle(self.overlay, COLOR_RAW_TARGET_RGBA, (x_px, y_px), RAW_TARGET_MARKER_RADIUS_PX)

            if self.radar.show_tracks:
                for tr in self.tracker.tracks:
                    x, y = tr.x[0], tr.x[1]
                    if math.hypot(x - self.cx, y - self.cy) > self.scope_r:
                        continue
                    color = COLOR_TRACK_CONFIRMED if tr.confirmed else COLOR_TRACK_TENTATIVE

                    if self.radar.show_trails and len(tr.history) > 1:
                        pts = [(int(px), int(py)) for (px, py) in tr.history]
                        pygame.draw.lines(self.overlay, COLOR_TRAIL_RGBA, False, pts, TRACK_TRAIL_THICKNESS_PX)

                    vx, vy = tr.x[2], tr.x[3]
                    speed_px = math.hypot(vx, vy)
                    vel_scale = clamp(speed_px / TARGET_SPEED_SOFT_MAX_DIVISOR, SWEEP_VEL_SCALE_MIN, SWEEP_VEL_SCALE_MAX)
                    pygame.draw.line(self.overlay, COLOR_VELOCITY_RGBA,
                                     (int(x), int(y)),
                                     (int(x + vx * vel_scale), int(y + vy * vel_scale)),
                                     VELOCITY_VECTOR_THICKNESS_PX)

                    pygame.draw.circle(self.overlay, color, (int(x), int(y)), TRACK_MARKER_OUTER_RADIUS_PX, GRID_LINE_THIN_PX+1)
                    pygame.draw.circle(self.overlay, color, (int(x), int(y)), TRACK_MARKER_INNER_RADIUS_PX)

                    if self.radar.show_labels:
                        speed_mps_est = speed_px * (self.radar.max_range_m / self.scope_r)
                        label_text = f"#{tr.id}{'✓' if tr.confirmed else ''}  {speed_mps_est:.0f} m/s"
                        lbl = FONT.render(label_text, True, COLOR_UI_TEXT)
                        self.overlay.blit(lbl, (int(x) + RAW_TARGET_MARKER_RADIUS_PX * 4, int(y) - UI_GRIDLINE_OFFSET_Y()))

            # Base clear
            self.screen.fill(COLOR_BACKGROUND)
            pygame.draw.circle(self.screen, COLOR_SCOPE_FRAME, (self.cx, self.cy), self.scope_r + SCOPE_FRAME_EXTRA_RADIUS_PX, SCOPE_FRAME_THICKNESS_PX)
            pygame.draw.circle(self.screen, COLOR_SCOPE_RING, (self.cx, self.cy), self.scope_r, SCOPE_RING_THICKNESS_PX)

            grid_surface = pygame.Surface(self.ppi.get_size(), pygame.SRCALPHA)
            self.draw_scope_grid(grid_surface)
            self.screen.blit(grid_surface, (0, 0))
            self.screen.blit(self.ppi, (0, 0))
            self.screen.blit(self.overlay, (0, 0))

            # UI panel
            pygame.draw.rect(self.screen, COLOR_UI_PANEL_BG, (self.W - self.panel_w, 0, self.panel_w, self.H))
            pygame.draw.rect(self.screen, COLOR_UI_PANEL_BORDER, (self.W - self.panel_w, 0, self.panel_w, self.H), GRID_LINE_THIN_PX+1)
            title = FONT_LG.render(UI_TITLE_TEXT, True, COLOR_UI_TEXT)
            self.screen.blit(title, (self.W - self.panel_w + UI_PANEL_PADDING_PX, GRID_LINE_THIN_PX * 4))

            for el in self.ui_elements:
                if hasattr(el, "draw"): el.draw(self.screen)

            help_text = FONT.render(UI_FOOTER_TEXT, True, COLOR_UI_TEXT_MUTED)
            self.screen.blit(help_text, (self.W - self.panel_w + UI_PANEL_PADDING_PX, self.H - UI_FOOTER_MARGIN_BOTTOM_PX))

            stats_text = FONT.render(f"{UI_AZ_TRACKS_PREFIX}{self.radar.angle_deg:6.2f}°   Tracks: {len(self.tracker.tracks)}", True, COLOR_UI_STATS_TEXT)
            self.screen.blit(stats_text, (UI_STATS_MARGIN_LEFT_PX, self.H - UI_STATS_MARGIN_BOTTOM_PX))

            pygame.display.flip()

def UI_GRIDLINE_OFFSET_Y() -> int:
    return RING_LABEL_OFFSET_Y_PX + GRID_LINE_THIN_PX * 8

if __name__ == "__main__":
    App().run()
