import os, sys
import pygame
import numpy as np
from typing import Tuple
from constants import OBSTACLE, START, GOAL, TRAIL, QUICKSAND, QS_STEPPED
from constants import UP, DOWN, LEFT, RIGHT

def _clamp(x: float) -> int:
    return max(0, min(255, int(x)))


def _brighten(rgb: Tuple[int, int, int], k: float = 1.25) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (_clamp(r * k), _clamp(g * k), _clamp(b * k))


# ── Mode selection menu ──────────────────────────────────────────────
def _show_menu(screen, W, H):
    """Show a mode selection screen. Returns 'manual' or 'auto'."""
    BG = (18, 18, 28)
    TITLE_COLOR = (0, 230, 180)
    BTN_COLOR = (40, 40, 65)
    BTN_HOVER = (60, 60, 100)
    BTN_TEXT = (230, 230, 230)
    HINT_COLOR = (140, 140, 160)

    title_font = pygame.font.SysFont("Arial", 48, bold=True)
    btn_font = pygame.font.SysFont("Arial", 32)
    hint_font = pygame.font.SysFont("Arial", 18)

    btn_w, btn_h = 300, 70
    spacing = 40
    total_h = btn_h * 2 + spacing
    top_y = H // 2 - total_h // 2 + 30

    btn_manual = pygame.Rect(W // 2 - btn_w // 2, top_y, btn_w, btn_h)
    btn_auto = pygame.Rect(W // 2 - btn_w // 2, top_y + btn_h + spacing, btn_w, btn_h)

    while True:
        mx, my = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if btn_manual.collidepoint(mx, my):
                    return "manual"
                if btn_auto.collidepoint(mx, my):
                    return "auto"

        screen.fill(BG)

        # title
        title_surf = title_font.render("GridWorld Q-Learning", True, TITLE_COLOR)
        screen.blit(title_surf, (W // 2 - title_surf.get_width() // 2, top_y - 110))

        subtitle_surf = hint_font.render("Select a play mode to begin", True, HINT_COLOR)
        screen.blit(subtitle_surf, (W // 2 - subtitle_surf.get_width() // 2, top_y - 50))

        # manual button
        hover_m = btn_manual.collidepoint(mx, my)
        pygame.draw.rect(screen, BTN_HOVER if hover_m else BTN_COLOR, btn_manual, border_radius=12)
        pygame.draw.rect(screen, TITLE_COLOR, btn_manual, width=2, border_radius=12)
        m_text = btn_font.render("Manual  (WASD)", True, BTN_TEXT)
        screen.blit(m_text, (btn_manual.centerx - m_text.get_width() // 2,
                             btn_manual.centery - m_text.get_height() // 2))

        # auto button
        hover_a = btn_auto.collidepoint(mx, my)
        pygame.draw.rect(screen, BTN_HOVER if hover_a else BTN_COLOR, btn_auto, border_radius=12)
        pygame.draw.rect(screen, TITLE_COLOR, btn_auto, width=2, border_radius=12)
        a_text = btn_font.render("Auto  (Q-Policy)", True, BTN_TEXT)
        screen.blit(a_text, (btn_auto.centerx - a_text.get_width() // 2,
                             btn_auto.centery - a_text.get_height() // 2))

        # hints
        hints = [
            "Manual: W ↑  A ←  S ↓  D →    R = restart    ESC = quit",
            "Auto:  Agent follows learned Q-policy automatically",
        ]
        for i, h in enumerate(hints):
            hs = hint_font.render(h, True, HINT_COLOR)
            screen.blit(hs, (W // 2 - hs.get_width() // 2,
                             btn_auto.bottom + 30 + i * 26))

        pygame.display.flip()
        pygame.time.Clock().tick(30)


# ── Main viewer ──────────────────────────────────────────────────────
def run_policy_pygame(
    base_grid: np.ndarray,
    meta,
    env,
    learner,
    *,
    window: Tuple[int, int] = (900, 900),
    pad: int = 14,
    gap: int = 2,
    radius: int = 6,
    fps: int = 30,
    step_every_n_frames: int = 6,
    greedy: bool = True,
):
    pygame.init()
    W, H = window
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(meta.title + " | Select Mode")
    clock = pygame.time.Clock()

    # ── setup ──
    font = pygame.font.SysFont("Arial", 16)
    hint_font = pygame.font.SysFont("Arial", 14)

    rows, cols = meta.rows, meta.cols
    tile = min((W - 2 * pad) // cols, (H - 2 * pad) // rows)
    if tile <= gap + 2:
        raise ValueError("Window too small for given pad/gap; increase window size or reduce pad/gap.")

    # WASD → action mapping
    KEY_TO_ACTION = {
        pygame.K_w: UP,
        pygame.K_a: LEFT,
        pygame.K_s: DOWN,
        pygame.K_d: RIGHT,
    }

    # ── outer loop: menu → play → M to go back ──
    while True:
        mode = _show_menu(screen, W, H)

        caption = meta.title + (" | Manual (WASD)" if mode == "manual" else " | Auto (Greedy Replay)")
        pygame.display.set_caption(caption)

        s = env.reset()
        a = 0
        if mode == "auto":
            a = learner.greedy_action(s) if greedy else learner.querysetstate(s)
        frame = 0
        done = False
        back_to_menu = False

        # ── game loop ──
        while not back_to_menu:
            # ── events ──
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if e.key == pygame.K_m:
                        back_to_menu = True
                    if e.key == pygame.K_r:
                        s = env.reset()
                        a = 0
                        if mode == "auto":
                            a = learner.greedy_action(s) if greedy else learner.querysetstate(s)
                        frame = 0
                        done = False
                    if mode == "manual" and not done and e.key in KEY_TO_ACTION:
                        action = KEY_TO_ACTION[e.key]
                        s_prime, r, done, _ = env.step(action)
                        s = s_prime

            if back_to_menu:
                break

            # ── auto step ──
            if mode == "auto" and (not done) and (frame % step_every_n_frames == 0):
                s_prime, r, done, _ = env.step(a)
                if greedy:
                    a = learner.greedy_action(s_prime)
                else:
                    a = learner.query(s_prime, r)
                s = s_prime

            # ── draw checker background ──
            for rr in range(rows):
                for cc in range(cols):
                    x = pad + cc * tile
                    y = pad + rr * tile
                    outer = pygame.Rect(x, y, tile, tile)
                    bg = meta.checker_a if (rr + cc) % 2 == 0 else meta.checker_b
                    pygame.draw.rect(screen, bg, outer)

            # ── draw tiles ──
            for rr in range(rows):
                for cc in range(cols):
                    v = int(base_grid[rr, cc])
                    x = pad + cc * tile
                    y = pad + rr * tile

                    outer = pygame.Rect(x, y, tile, tile)
                    pygame.draw.rect(screen, meta.grid_line, outer, width=1, border_radius=radius)

                    inner = pygame.Rect(x + gap // 2, y + gap // 2, tile - gap, tile - gap)
                    color = meta.palette.get(v, (200, 0, 200))

                    if meta.title.lower().startswith("neon") and v in (OBSTACLE, START, GOAL, QUICKSAND, QS_STEPPED):
                        pygame.draw.rect(screen, _brighten(color, 1.35), inner, border_radius=radius)
                        inner2 = inner.inflate(-2, -2)
                        pygame.draw.rect(screen, color, inner2, border_radius=max(4, radius - 3))
                    else:
                        pygame.draw.rect(screen, color, inner, border_radius=radius)

                    # start / goal mark
                    if v == START:
                        pygame.draw.circle(screen, (255, 255, 255), inner.center, max(6, tile // 6))
                    elif v == GOAL:
                        pygame.draw.circle(screen, (255, 255, 255), inner.center, max(6, tile // 6), width=3)

            # ── draw trail overlay ──
            trail_color = meta.palette.get(TRAIL, (120, 120, 140))
            for (rr, cc) in env.trail:
                x = pad + cc * tile
                y = pad + rr * tile
                inner = pygame.Rect(x + gap // 2 + 8, y + gap // 2 + 8, tile - gap - 16, tile - gap - 16)
                pygame.draw.rect(screen, trail_color, inner, border_radius=max(4, radius - 6))

            # ── draw agent ──
            ar, ac = env.pos
            ax = pad + ac * tile
            ay = pad + ar * tile
            inner = pygame.Rect(ax + gap // 2, ay + gap // 2, tile - gap, tile - gap)
            pygame.draw.circle(screen, (255, 255, 255), inner.center, max(6, tile // 5))

            # ── HUD ──
            if mode == "manual":
                hud_text = f"MANUAL | steps: {env.steps} | done: {done} | pos: {env.pos} | M=menu  R=restart  ESC=quit"
            else:
                hud_text = f"AUTO | steps: {env.steps} | done: {done} | pos: {env.pos} | M=menu  R=restart  ESC=quit"
            hud = font.render(hud_text, True, (220, 220, 220))
            screen.blit(hud, (10, 10))

            # control hints for manual mode
            if mode == "manual" and not done:
                hint = hint_font.render("W ↑   A ←   S ↓   D →", True, (100, 220, 180))
                screen.blit(hint, (W - hint.get_width() - 14, H - 26))

            if done:
                done_surf = font.render("GOAL REACHED!" if env.pos == meta.goal else "MAX STEPS!", True, (255, 220, 60))
                screen.blit(done_surf, (W // 2 - done_surf.get_width() // 2, H - 30))

            pygame.display.flip()
            clock.tick(fps)
            frame += 1
