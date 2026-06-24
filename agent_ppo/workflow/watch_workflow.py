from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from agent_ppo.conf.conf import Config
from agent_ppo.workflow.evaluate_workflow import ObsNormalizer, _make_env, _model_path, _vecnormalize_path


PANEL_WIDTH = 360
WINDOW_PADDING = 18
BUTTON_H = 36
BUTTON_GAP = 10
FPS = 30
STEP_INTERVAL_MS = 120

BG = (232, 234, 238)
PANEL_BG = (246, 247, 249)
PANEL_BORDER = (205, 210, 218)
TEXT = (30, 34, 42)
TEXT_MUTED = (92, 99, 112)
BUTTON_BG = (230, 234, 242)
BUTTON_ACTIVE = (214, 225, 255)
BUTTON_TEXT = (25, 29, 36)
GOOD = (31, 150, 74)
BAD = (190, 55, 60)


class Button:
    def __init__(self, rect, label: str):
        self.rect = rect
        self.label = label

    def draw(self, surface, font, active: bool = False):
        import pygame

        bg = BUTTON_ACTIVE if active else BUTTON_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 1, border_radius=6)
        text = font.render(self.label, True, BUTTON_TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


class MiniGridViewer:
    def __init__(self, exp_id: int, n_timesteps: int | None = None):
        import pygame

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("microsoftyahei,simhei,arial", 18)
        self.small_font = pygame.font.SysFont("microsoftyahei,simhei,arial", 14)
        self.title_font = pygame.font.SysFont("microsoftyahei,simhei,arial", 22, bold=True)

        self.exp_id = exp_id
        self.max_steps = int(n_timesteps or 1000)
        self.model_path = _model_path(exp_id)
        self.normalizer = ObsNormalizer(_vecnormalize_path(exp_id))
        self.model = _load_model(self.model_path)
        self.env = _make_env(render_mode="rgb_array")

        self.obs = None
        self.done = False
        self.auto_run = False
        self.step_count = 0
        self.episode = 1
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.status = "Ready"
        self.last_step_time = 0

        self.frame = self._render_frame_after_reset()
        frame_h, frame_w = self.frame.shape[:2]
        self.scale = max(3, min(6, 720 // max(frame_h, 1)))
        self.view_w = frame_w * self.scale
        self.view_h = frame_h * self.scale
        self.width = PANEL_WIDTH + self.view_w + WINDOW_PADDING * 3
        self.height = max(self.view_h + WINDOW_PADDING * 2, 560)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MiniGrid PPO Viewer")
        self.clock = pygame.time.Clock()
        self._build_layout()

    def _build_layout(self):
        import pygame

        x = WINDOW_PADDING
        panel_w = PANEL_WIDTH - WINDOW_PADDING * 2
        half = (panel_w - BUTTON_GAP) // 2
        self.buttons = {
            "run": Button(pygame.Rect(x, 146, half, BUTTON_H), "开始"),
            "pause": Button(pygame.Rect(x + half + BUTTON_GAP, 146, half, BUTTON_H), "暂停"),
            "step": Button(pygame.Rect(x, 192, half, BUTTON_H), "单步"),
            "reset": Button(pygame.Rect(x + half + BUTTON_GAP, 192, half, BUTTON_H), "重置"),
            "quit": Button(pygame.Rect(x, 238, panel_w, BUTTON_H), "退出"),
        }

    def _render_frame_after_reset(self):
        reset_out = self.env.reset()
        self.obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        frame = self.env.render()
        return _ensure_rgb_frame(frame)

    def reset_episode(self):
        self.done = False
        self.auto_run = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.status = "Reset"
        self.frame = self._render_frame_after_reset()

    def step_agent(self):
        if self.done:
            return
        if self.step_count >= self.max_steps:
            self.done = True
            self.auto_run = False
            self.status = "Reached max timesteps"
            return

        norm_obs = self.normalizer.normalize(self.obs)[None, :]
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        step_out = self.env.step(action)
        if len(step_out) == 5:
            self.obs, reward, terminated, truncated, _info = step_out
            self.done = terminated or truncated
        else:
            self.obs, reward, self.done, _info = step_out

        self.last_reward = float(reward)
        self.episode_reward += self.last_reward
        self.step_count += 1
        self.frame = _ensure_rgb_frame(self.env.render())
        self.status = f"Action {action}"
        if self.done:
            self.auto_run = False
            self.status = "Episode finished"

    def _draw(self):
        import pygame

        self.screen.fill(BG)
        self._draw_minigrid()
        self._draw_panel()
        pygame.display.flip()

    def _draw_minigrid(self):
        import pygame

        view_left = PANEL_WIDTH + WINDOW_PADDING * 2
        view_top = WINDOW_PADDING
        surf = pygame.surfarray.make_surface(np.transpose(self.frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (self.view_w, self.view_h))
        self.screen.blit(surf, (view_left, view_top))
        pygame.draw.rect(self.screen, PANEL_BORDER, (view_left, view_top, self.view_w, self.view_h), 1)

    def _draw_panel(self):
        import pygame

        panel = pygame.Rect(WINDOW_PADDING, WINDOW_PADDING, PANEL_WIDTH - WINDOW_PADDING * 2, self.height - WINDOW_PADDING * 2)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 1)

        x = WINDOW_PADDING + 16
        y = WINDOW_PADDING + 18
        self.screen.blit(self.title_font.render("MiniGrid PPO Viewer", True, TEXT), (x, y))
        y += 38
        self.screen.blit(self.small_font.render(f"Env: {Config.ENV_ID}", True, TEXT_MUTED), (x, y))
        y += 24
        self.screen.blit(self.small_font.render(f"Exp ID: {self.exp_id}", True, TEXT_MUTED), (x, y))
        y += 24
        self.screen.blit(self.small_font.render(f"Model: {self.model_path.name}", True, TEXT_MUTED), (x, y))

        self.buttons["run"].draw(self.screen, self.font, active=self.auto_run)
        self.buttons["pause"].draw(self.screen, self.font, active=not self.auto_run)
        self.buttons["step"].draw(self.screen, self.font)
        self.buttons["reset"].draw(self.screen, self.font)
        self.buttons["quit"].draw(self.screen, self.font)

        y = 304
        status_color = GOOD if self.episode_reward > 0 else (BAD if self.done else TEXT)
        lines = [
            f"状态: {self.status}",
            f"回合: {self.episode}",
            f"步数: {self.step_count}/{self.max_steps}",
            f"本步奖励: {self.last_reward:.3f}",
            f"累计奖励: {self.episode_reward:.3f}",
            f"是否结束: {self.done}",
            "",
            "快捷键:",
            "Space 开始/暂停",
            "S 单步",
            "R 重置",
            "Esc 退出",
        ]
        for line in lines:
            color = status_color if line.startswith("状态:") else TEXT
            self.screen.blit(self.small_font.render(line, True, color), (x, y))
            y += 24

    def _handle_click(self, pos):
        if self.buttons["run"].hit(pos):
            self.auto_run = True
            self.status = "Running"
        elif self.buttons["pause"].hit(pos):
            self.auto_run = False
            self.status = "Paused"
        elif self.buttons["step"].hit(pos):
            self.step_agent()
        elif self.buttons["reset"].hit(pos):
            self.reset_episode()
        elif self.buttons["quit"].hit(pos):
            return False
        return True

    def _handle_key(self, key):
        import pygame

        if key == pygame.K_SPACE:
            self.auto_run = not self.auto_run
            self.status = "Running" if self.auto_run else "Paused"
        elif key == pygame.K_s:
            self.step_agent()
        elif key == pygame.K_r:
            self.reset_episode()
        elif key == pygame.K_ESCAPE:
            return False
        return True

    def run(self):
        import pygame

        running = True
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    running = self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)

            if self.auto_run and now - self.last_step_time >= STEP_INTERVAL_MS:
                self.step_agent()
                self.last_step_time = now

            self._draw()
            self.clock.tick(FPS)

        self.env.close()
        pygame.quit()


def _load_model(model_path: Path):
    return PPO.load(
        str(model_path),
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
        device="cpu",
    )


def _ensure_rgb_frame(frame):
    if frame is None:
        raise RuntimeError("MiniGrid did not return an rgb_array frame. Create the environment with render_mode='rgb_array'.")
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unsupported render frame shape: {frame.shape}")
    return frame.astype(np.uint8)


def _run_no_render(exp_id: int, total_steps: int) -> None:
    model_path = _model_path(exp_id)
    normalizer = ObsNormalizer(_vecnormalize_path(exp_id))
    model = _load_model(model_path)
    env = _make_env()
    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        episode_reward = 0.0
        episode_len = 0
        episode = 1

        print(f"model: {model_path}")
        print(f"timesteps: {total_steps}")

        for _ in range(total_steps):
            norm_obs = normalizer.normalize(obs)[None, :]
            action, _ = model.predict(norm_obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, _info = step_out

            episode_reward += float(reward)
            episode_len += 1
            if done:
                print(f"episode {episode}: reward={episode_reward:.3f}, length={episode_len}")
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                episode_reward = 0.0
                episode_len = 0
                episode += 1
    finally:
        env.close()


def workflow(
    no_render: bool = False,
    n_timesteps: int | None = None,
    exp_id: int | None = Config.EVAL_EXP_ID,
) -> None:
    total_steps = int(n_timesteps or 1000)
    run_id = int(exp_id or Config.EVAL_EXP_ID)
    if no_render:
        _run_no_render(run_id, total_steps)
        return
    MiniGridViewer(run_id, n_timesteps=total_steps).run()
