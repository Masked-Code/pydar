import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import config
from controls import Slider, Toggle, Button
from core import wrap_angle_deg, lerp, pol2cart, cart2pol, clamp, init_fonts

import numpy as np
import pygame

fonts = init_fonts()
FONT = fonts['default']
FONT_SM = fonts['small']
FONT_LG = fonts['large']


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
            self.maneuver_timer = random.uniform(config.TARGET_MANEUVER_MIN_S, config.TARGET_MANEUVER_MAX_S)
            ax = random.uniform(config.TARGET_ACCELERATION_MIN, config.TARGET_ACCELERATION_MAX)
            ay = random.uniform(config.TARGET_ACCELERATION_MIN, config.TARGET_ACCELERATION_MAX)
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
    history: deque = field(default_factory=lambda: deque(maxlen=config.TRACK_TRAIL_MAXLEN))
    color: Tuple[int,int,int] = field(default_factory=lambda: (random.randint(80,255), random.randint(150,255), 100))
    confirmed: bool = False

class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.sigma_pos = config.TRACKER_MEASUREMENT_STD_PX
        self.gate_px = config.TRACKER_ASSOCIATION_GATE_PX
        self.confirm_hits = config.TRACKER_CONFIRM_HITS_REQUIRED
        self.max_misses = config.TRACKER_MAX_MISSES_ALLOWED
        self.q_accel = config.TRACKER_PROCESS_NOISE_ACCEL

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
                config.TRACKER_INITIAL_POSITION_VAR,
                config.TRACKER_INITIAL_POSITION_VAR,
                config.TRACKER_INITIAL_VELOCITY_VAR,
                config.TRACKER_INITIAL_VELOCITY_VAR
            ])
            self.tracks.append(Track(self.next_id, x0, P0))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

class Radar:
    def __init__(self, scope_radius_px: int):
        self.scope_r_px = scope_radius_px
        self.max_range_m = config.RADAR_DEFAULT_MAX_RANGE_M
        self.range_res_m = config.RADAR_DEFAULT_RANGE_RES_M
        self.beamwidth_deg = config.RADAR_DEFAULT_BEAMWIDTH_DEG
        self.rpm = config.RADAR_DEFAULT_RPM
        self.angle_deg = 0.0

        self.noise_power = config.RADAR_DEFAULT_NOISE_POWER
        self.clutter_on = config.RADAR_DEFAULT_CLUTTER_ENABLED
        self.false_alarm_density = config.RADAR_DEFAULT_FALSE_ALARM_DENSITY

        self.cfar_guard = config.CFAR_DEFAULT_GUARD_CELLS
        self.cfar_train = config.CFAR_DEFAULT_TRAINING_CELLS
        self.cfar_scale = config.CFAR_DEFAULT_SCALE

        self.afterglow_decay = config.AFTERGLOW_DEFAULT_ALPHA_DECAY_PER_FRAME

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
        self.angle_deg = wrap_angle_deg(self.angle_deg + self.rpm * config.SPEED_RPM_TO_DEG_PER_SEC * dt)

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
            r = np.arange(nbins) * dr + config.GRID_LINE_THIN_PX 
            clutter_profile = self.clutter_base_scale * (1.0 / np.sqrt(r))
            azimuthal_sector_mod = self.clutter_sector_scale * (1.0 + math.sin((angle_deg * 3.0) * config.DEG_TO_RAD))
            returns += clutter_profile * azimuthal_sector_mod * np.random.randn(nbins)

        for tg in targets:
            rng_m, bearing_deg = cart2pol(tg.x, tg.y)
            if rng_m > Rmax:
                continue
            off = min(abs(wrap_angle_deg(bearing_deg - angle_deg)),
                      config.DEG_FULL_CIRCLE - abs(wrap_angle_deg(bearing_deg - angle_deg)))
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
        last_idx = - (config.CFAR_PLATEAU_SUPPRESS_DISTANCE_BINS + 10) 
        for idx in hits:
            if idx - last_idx > config.CFAR_PLATEAU_SUPPRESS_DISTANCE_BINS:
                pruned.append(idx)
            last_idx = idx
        return pruned

class App:
    def __init__(self):
        self.W, self.H = config.WINDOW_WIDTH_PX, config.WINDOW_HEIGHT_PX
        self.panel_w = config.CONTROL_PANEL_WIDTH_PX
        self.cx, self.cy = (self.W - self.panel_w) // 2, self.H // 2
        self.scope_r = min(self.cx, self.cy) - config.SCOPE_MARGIN_PX

        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("High-Fidelity Interactive Radar Simulation")
        self.clock = pygame.time.Clock()

        self.ppi = pygame.Surface((self.W - self.panel_w, self.H), pygame.SRCALPHA)
        self.overlay = pygame.Surface((self.W - self.panel_w, self.H), pygame.SRCALPHA)

        self.radar = Radar(self.scope_r)
        self.tracker = Tracker()

        self.targets: List[Target] = []
        self.target_count = config.TARGET_DEFAULT_COUNT
        self.target_speed_max = config.SLIDER_MAX_SPEED_MAX_MPS - (config.SLIDER_MAX_SPEED_MAX_MPS - config.SLIDER_MAX_SPEED_MIN_MPS) / 2.0
        self.spawn_targets(self.target_count)

        x0 = self.W - self.panel_w + config.UI_PANEL_PADDING_PX
        y = config.UI_PANEL_PADDING_PX
        w = self.panel_w - config.UI_PANEL_PADDING_PX * 2

        self.s_max_range = Slider(x0, y, w, config.SLIDER_LABEL_MAX_RANGE_KM,
                                  config.SLIDER_MAX_RANGE_KM_MIN, config.SLIDER_MAX_RANGE_KM_MAX,
                                  self.radar.max_range_m / 1000.0, step=config.SLIDER_MAX_RANGE_KM_STEP, fmt="{:.0f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_rpm       = Slider(x0, y, w, config.SLIDER_LABEL_RPM,
                                  config.SLIDER_RPM_MIN, config.SLIDER_RPM_MAX,
                                  self.radar.rpm, step=config.SLIDER_RPM_STEP, fmt="{:.0f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_bw        = Slider(x0, y, w, config.SLIDER_LABEL_BEAMWIDTH_DEG,
                                  config.SLIDER_BEAMWIDTH_MIN_DEG, config.SLIDER_BEAMWIDTH_MAX_DEG,
                                  self.radar.beamwidth_deg, step=config.SLIDER_BEAMWIDTH_STEP_DEG, fmt="{:.1f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_res       = Slider(x0, y, w, config.SLIDER_LABEL_RANGE_RES_M,
                                  config.SLIDER_RANGE_RES_MIN_M, config.SLIDER_RANGE_RES_MAX_M,
                                  self.radar.range_res_m, step=config.SLIDER_RANGE_RES_STEP_M, fmt="{:.0f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_noise     = Slider(x0, y, w, config.SLIDER_LABEL_NOISE,
                                  config.SLIDER_NOISE_MIN, config.SLIDER_NOISE_MAX,
                                  self.radar.noise_power, fmt="{:.2f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_cfar      = Slider(x0, y, w, config.SLIDER_LABEL_CFAR,
                                  config.SLIDER_CFAR_MIN, config.SLIDER_CFAR_MAX,
                                  self.radar.cfar_scale, step=config.SLIDER_CFAR_STEP, fmt="{:.1f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_after     = Slider(x0, y, w, config.SLIDER_LABEL_AFTERGLOW,
                                  config.SLIDER_AFTERGLOW_MIN, config.SLIDER_AFTERGLOW_MAX,
                                  self.radar.afterglow_decay, step=config.SLIDER_AFTERGLOW_STEP, fmt="{:.0f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_tgt_count = Slider(x0, y, w, config.SLIDER_LABEL_TARGETS,
                                  config.SLIDER_TARGETS_MIN, config.SLIDER_TARGETS_MAX,
                                  self.target_count, step=1.0, fmt="{:.0f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX
        self.s_tgt_spd   = Slider(x0, y, w, config.SLIDER_LABEL_MAX_SPEED,
                                  config.SLIDER_MAX_SPEED_MIN_MPS, config.SLIDER_MAX_SPEED_MAX_MPS,
                                  self.target_speed_max, step=config.SLIDER_MAX_SPEED_STEP_MPS, fmt="{:.0f}", font=FONT); y += config.UI_VERTICAL_SPACING_PX

        y += config.GRID_LINE_THIN_PX * 8
        self.t_show_raw   = Toggle(x0, y, config.LABEL_SHOW_RAW, False, font=FONT); y += config.UI_TOGGLE_SPACING_PX
        self.t_clutter    = Toggle(x0, y, config.LABEL_CLUTTER, True, font=FONT); y += config.UI_TOGGLE_SPACING_PX
        self.t_trails     = Toggle(x0, y, config.LABEL_TRAILS, True, font=FONT); y += config.UI_TOGGLE_SPACING_PX
        self.t_labels     = Toggle(x0, y, config.LABEL_LABELS, True, font=FONT); y += config.UI_TOGGLE_EXTRA_GAP_PX

        half_w = (w - config.UI_BUTTON_COL_GAP_PX) // 2
        self.b_add_tgt    = Button(x0, y, half_w, config.UI_BUTTON_HEIGHT_PX, config.LABEL_SPAWN_BUTTON, font=FONT)
        self.b_rem_tgt    = Button(x0 + half_w + config.UI_BUTTON_COL_GAP_PX, y, half_w, config.UI_BUTTON_HEIGHT_PX, config.LABEL_REMOVE_BUTTON, font=FONT); y += config.UI_BUTTON_ROW_GAP_PX
        self.b_reset_tr   = Button(x0, y, w, config.UI_BUTTON_HEIGHT_PX, config.LABEL_RESET_TRACKS, font=FONT); y += config.UI_RESET_GAP_PX
        self.b_pause      = Button(x0, y, w, config.UI_BUTTON_HEIGHT_PX, config.LABEL_PAUSE_STEP, font=FONT); y += config.UI_RESET_GAP_PX

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
            r = random.uniform(config.TARGET_RANGE_SPAWN_MIN_M, self.radar.max_range_m * config.TARGET_RANGE_SPAWN_BULK_MAX_FACTOR)
            th = random.uniform(0.0, config.DEG_FULL_CIRCLE)
            x, y = pol2cart(r, th)
            speed = random.uniform(self.target_speed_max * config.TARGET_SPEED_MIN_FACTOR, self.target_speed_max)
            heading = random.uniform(0.0, config.DEG_FULL_CIRCLE)
            vx, vy = pol2cart(speed, heading)
            rcs = 10 ** random.uniform(config.TARGET_RCS_DECADE_MIN, config.TARGET_RCS_DECADE_MAX)
            self.targets.append(Target(x, y, vx, vy, rcs))

    def add_one_target(self):
        r = random.uniform(config.TARGET_RANGE_SPAWN_INNER_MIN_M, self.radar.max_range_m * config.TARGET_RANGE_SPAWN_MAX_FACTOR)
        th = random.uniform(0.0, config.DEG_FULL_CIRCLE)
        x, y = pol2cart(r, th)
        speed = random.uniform(self.target_speed_max * config.TARGET_SPEED_SPAWN_MIN_FACTOR, self.target_speed_max)
        heading = random.uniform(0.0, config.DEG_FULL_CIRCLE)
        vx, vy = pol2cart(speed, heading)
        rcs = 10 ** random.uniform(config.TARGET_RCS_DECADE_MIN, config.TARGET_RCS_DECADE_MAX)
        self.targets.append(Target(x, y, vx, vy, rcs))

    def remove_one_target(self):
        if self.targets:
            self.targets.pop(random.randrange(len(self.targets)))

    def draw_scope_grid(self, surf):
        cx, cy, R = self.cx, self.cy, self.scope_r
        for i in range(1, config.RING_COUNT + 1):
            r = int(R * i / config.RING_COUNT)
            pygame.draw.circle(surf, config.COLOR_GRID_RING, (cx, cy), r, config.GRID_LINE_THIN_PX)
            rng_km = self.radar.max_range_m * (i / config.RING_COUNT) / 1000.0
            lbl = FONT_SM.render(f"{rng_km:.0f} km", True, config.COLOR_RING_TEXT)
            surf.blit(lbl, (cx + r + config.RING_LABEL_OFFSET_X_PX, cy - config.RING_LABEL_OFFSET_Y_PX))
        pygame.draw.line(surf, config.COLOR_GRID_CROSS, (cx - R, cy), (cx + R, cy), config.GRID_LINE_THIN_PX)
        pygame.draw.line(surf, config.COLOR_GRID_CROSS, (cx, cy - R), (cx, cy + R), config.GRID_LINE_THIN_PX)
        for deg in range(0, int(config.DEG_FULL_CIRCLE), config.AZIMUTH_TICK_STEP_DEG):
            x1, y1 = pol2cart(R - config.AZIMUTH_TICK_INSET_PX, deg)
            x2, y2 = pol2cart(R, deg)
            pygame.draw.line(surf, config.COLOR_AZ_TICK, (cx + int(x1), cy + int(y1)), (cx + int(x2), cy + int(y2)), config.GRID_LINE_THIN_PX)

    def fade_afterglow(self):
        decay = int(self.radar.afterglow_decay)
        decay = clamp(decay, config.MIN_AFTERGLOW_ALPHA, config.MAX_AFTERGLOW_ALPHA)
        if decay <= 0:
            return
        fade = pygame.Surface(self.ppi.get_size(), pygame.SRCALPHA)
        fade.fill((0, 0, 0, int(decay)))
        self.ppi.blit(fade, (0, 0), special_flags=config.AFTERGLOW_BLEND_MODE)

    def draw_sweep(self, angle_deg):
        bw = self.radar.beamwidth_deg
        R = self.scope_r
        cx, cy = self.cx, self.cy
        left = wrap_angle_deg(angle_deg - bw / 2.0)
        right = wrap_angle_deg(angle_deg + bw / 2.0)
        pts = [(cx, cy)]
        for t in np.linspace(left, right, config.SWEEP_POLY_SEGMENTS):
            x, y = pol2cart(R, t)
            pts.append((cx + int(x), cy + int(y)))
        pygame.draw.polygon(self.overlay, config.COLOR_SWEEP_WEDGE_RGBA, pts)
        hx, hy = pol2cart(R, angle_deg)
        pygame.draw.line(self.overlay, config.COLOR_SWEEP_HEAD_RGBA, (cx, cy), (cx + int(hx), cy + int(hy)), config.SWEEP_HEAD_THICKNESS_PX)

    def draw_blip(self, x_px, y_px, strong=True):
        core_alpha = config.COLOR_BLIP_CORE_RGBA[3] if strong else int(config.COLOR_BLIP_CORE_RGBA[3] * 0.52)
        core_color = (config.COLOR_BLIP_CORE_RGBA[0], config.COLOR_BLIP_CORE_RGBA[1], config.COLOR_BLIP_CORE_RGBA[2], core_alpha)
        pygame.draw.circle(self.ppi, core_color, (x_px, y_px), config.BLIP_CORE_RADIUS_PX)
        pygame.draw.circle(self.ppi, config.COLOR_BLIP_BLOOM_RGBA, (x_px, y_px), config.BLIP_BLOOM_RADIUS_PX, config.GRID_LINE_THIN_PX+1)

    def world_to_px(self, x_m, y_m):
        scale = self.radar.px_per_meter()
        return int(self.cx + x_m * scale), int(self.cy + y_m * scale)

    def run(self):
        while True:
            dt = self.clock.tick(config.FRAME_RATE_TARGET_FPS) / config.MS_PER_SECOND
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
                    rng += np.random.normal(0.0, self.radar.range_res_m * config.MEASUREMENT_RANGE_JITTER_STD_FACTOR)
                    ang = self.radar.angle_deg + np.random.normal(0.0, self.radar.beamwidth_deg * config.MEASUREMENT_ANGLE_JITTER_STD_FACTOR)
                    x_m, y_m = pol2cart(rng, ang)
                    x_px, y_px = self.world_to_px(x_m, y_m)
                    if math.hypot(x_px - self.cx, y_px - self.cy) <= self.scope_r:
                        self.draw_blip(x_px, y_px, strong=True)
                        det_xy.append(np.array([x_px, y_px], dtype=float))

                self.tracker.predict(dt)
                self.tracker.update(det_xy)
                self.fade_afterglow()

            self.overlay.fill((0, 0, 0, 0))
            self.draw_sweep(self.radar.angle_deg)

            if self.radar.show_raw:
                for t in self.targets:
                    x_px, y_px = self.world_to_px(t.x, t.y)
                    if math.hypot(x_px - self.cx, y_px - self.cy) <= self.scope_r:
                        pygame.draw.circle(self.overlay, config.COLOR_RAW_TARGET_RGBA, (x_px, y_px), config.RAW_TARGET_MARKER_RADIUS_PX)

            if self.radar.show_tracks:
                for tr in self.tracker.tracks:
                    x, y = tr.x[0], tr.x[1]
                    if math.hypot(x - self.cx, y - self.cy) > self.scope_r:
                        continue
                    color = config.COLOR_TRACK_CONFIRMED if tr.confirmed else config.COLOR_TRACK_TENTATIVE

                    if self.radar.show_trails and len(tr.history) > 1:
                        pts = [(int(px), int(py)) for (px, py) in tr.history]
                        pygame.draw.lines(self.overlay, config.COLOR_TRAIL_RGBA, False, pts, config.TRACK_TRAIL_THICKNESS_PX)

                    vx, vy = tr.x[2], tr.x[3]
                    speed_px = math.hypot(vx, vy)
                    vel_scale = clamp(speed_px / config.TARGET_SPEED_SOFT_MAX_DIVISOR, config.SWEEP_VEL_SCALE_MIN, config.SWEEP_VEL_SCALE_MAX)
                    pygame.draw.line(self.overlay, config.COLOR_VELOCITY_RGBA,
                                     (int(x), int(y)),
                                     (int(x + vx * vel_scale), int(y + vy * vel_scale)),
                                     config.VELOCITY_VECTOR_THICKNESS_PX)

                    pygame.draw.circle(self.overlay, color, (int(x), int(y)), config.TRACK_MARKER_OUTER_RADIUS_PX, config.GRID_LINE_THIN_PX+1)
                    pygame.draw.circle(self.overlay, color, (int(x), int(y)), config.TRACK_MARKER_INNER_RADIUS_PX)

                    if self.radar.show_labels:
                        speed_mps_est = speed_px * (self.radar.max_range_m / self.scope_r)
                        label_text = f"#{tr.id}{'✓' if tr.confirmed else ''}  {speed_mps_est:.0f} m/s"
                        lbl = FONT.render(label_text, True, config.COLOR_UI_TEXT)
                        self.overlay.blit(lbl, (int(x) + config.RAW_TARGET_MARKER_RADIUS_PX * 4, int(y) - UI_GRIDLINE_OFFSET_Y()))

            self.screen.fill(config.COLOR_BACKGROUND)
            pygame.draw.circle(self.screen, config.COLOR_SCOPE_FRAME, (self.cx, self.cy), self.scope_r + config.SCOPE_FRAME_EXTRA_RADIUS_PX, config.SCOPE_FRAME_THICKNESS_PX)
            pygame.draw.circle(self.screen, config.COLOR_SCOPE_RING, (self.cx, self.cy), self.scope_r, config.SCOPE_RING_THICKNESS_PX)

            grid_surface = pygame.Surface(self.ppi.get_size(), pygame.SRCALPHA)
            self.draw_scope_grid(grid_surface)
            self.screen.blit(grid_surface, (0, 0))
            self.screen.blit(self.ppi, (0, 0))
            self.screen.blit(self.overlay, (0, 0))

            pygame.draw.rect(self.screen, config.COLOR_UI_PANEL_BG, (self.W - self.panel_w, 0, self.panel_w, self.H))
            pygame.draw.rect(self.screen, config.COLOR_UI_PANEL_BORDER, (self.W - self.panel_w, 0, self.panel_w, self.H), config.GRID_LINE_THIN_PX+1)
            title = FONT_LG.render(config.UI_TITLE_TEXT, True, config.COLOR_UI_TEXT)
            self.screen.blit(title, (self.W - self.panel_w + config.UI_PANEL_PADDING_PX, config.GRID_LINE_THIN_PX * 4))

            for el in self.ui_elements:
                if hasattr(el, "draw"): el.draw(self.screen)

            help_text = FONT.render(config.UI_FOOTER_TEXT, True, config.COLOR_UI_TEXT_MUTED)
            self.screen.blit(help_text, (self.W - self.panel_w + config.UI_PANEL_PADDING_PX, self.H - config.UI_FOOTER_MARGIN_BOTTOM_PX))

            stats_text = FONT.render(f"{config.UI_AZ_TRACKS_PREFIX}{self.radar.angle_deg:6.2f}°   Tracks: {len(self.tracker.tracks)}", True, config.COLOR_UI_STATS_TEXT)
            self.screen.blit(stats_text, (config.UI_STATS_MARGIN_LEFT_PX, self.H - config.UI_STATS_MARGIN_BOTTOM_PX))

            pygame.display.flip()

def UI_GRIDLINE_OFFSET_Y() -> int:
    return config.RING_LABEL_OFFSET_Y_PX + config.GRID_LINE_THIN_PX * 8

if __name__ == "__main__":
    App().run()
