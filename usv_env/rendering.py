from __future__ import annotations

import importlib
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import rotate
from .types import ShipState

BLUE_SHIP_FILL = (35, 112, 210)
BLUE_SHIP_OUTLINE = (13, 55, 120)
BLUE_SHIP_2_FILL = (20, 160, 210)
BLUE_SHIP_2_OUTLINE = (5, 92, 126)
RED_SHIP_FILL = (218, 54, 54)
RED_SHIP_OUTLINE = (118, 24, 24)
SHADOW_RED_COLOR = (125, 125, 125)
SUBMERGED_SUBMARINE_FILL = (236, 196, 74)
SUCCESS_SUBMARINE_FILL = (45, 139, 87)
SUBMARINE_OUTLINE = (74, 82, 64)
PROTECTED_ZONE_OUTLINE = (3, 83, 80)
ROUTE_COLOR = (202, 96, 96)
ISLAND_FILL = (91, 166, 92)
ISLAND_OUTLINE = (44, 111, 63)
WORLD_BACKGROUND_FILL = (255, 255, 255)
GRID_LINE_COLOR = (226, 226, 226)
AXIS_LINE_COLOR = (98, 98, 98)
AXIS_TEXT_COLOR = (70, 70, 70)
PHYSICAL_READY_COLOR = (230, 151, 36)
# The task map is the original scene at 1:10 scale; retain the old visual
# density (roughly one grid line per 50 m in the original map).
GRID_STEP_METERS = 5.0


@dataclass
class RenderFrame:
    """Read-only rendering snapshot for the dynamic predictive task.

    Rendering receives only task state and never feeds visual state back into the
    environment. In particular, success highlighting does not alter ``ShipState``.
    """

    world_size: np.ndarray
    island_center: np.ndarray
    island_radius: float
    red_state: ShipState
    blue_states: list[ShipState]
    submarine_states: list[ShipState]
    route_start: np.ndarray
    route_end: np.ndarray
    phase: str
    physical_capture_success: bool
    cooperative_task_success: bool
    semi_major: float
    semi_minor: float
    surface_active_mask: np.ndarray | None = None
    submarine_active_mask: np.ndarray | None = None
    island_active: bool = True
    task_name: str = "final_task"
    difficulty_band: str | None = None
    # Formal pre-training diagnostics.  These fields are intentionally optional so
    # historical render snapshots remain loadable.
    shadow_red_state: ShipState | None = None
    show_shadow_red: bool = False
    event_markers: dict[str, bool] | None = None
    visualization_metadata: dict[str, Any] | None = None
    diagnostics: dict[str, float] | None = None
    simulation_time: float = 0.0
    termination_reason: str = ""
    reefs: list[dict] | None = None
    protected_center: np.ndarray | None = None
    protected_radius: float = 0.0
    encirclement_valid: bool = False
    red_behavior: str = ''


def submarine_fill_color(
    state: ShipState,
    cooperative_task_success: bool = False,
) -> tuple[int, int, int]:
    """Return a visual color without treating success as a physical state change."""

    del state
    return (
        SUCCESS_SUBMARINE_FILL
        if cooperative_task_success
        else SUBMERGED_SUBMARINE_FILL
    )


def axis_tick_values(limit: float, step: float = GRID_STEP_METERS) -> list[float]:
    if not np.isfinite(limit) or not np.isfinite(step) or step <= 0.0:
        return []
    count = int(math.floor(max(0.0, float(limit)) / float(step)))
    return [float(index) * float(step) for index in range(count + 1)]


def submarine_world_ellipse_points(
    state: ShipState,
    semi_major: float,
    semi_minor: float,
    point_count: int = 72,
) -> list[np.ndarray]:
    major = float(semi_major) * 1.05
    minor = float(semi_minor) * 0.72
    points = []
    for theta in np.linspace(0.0, 2.0 * math.pi, point_count, endpoint=False):
        local = np.array([major * math.cos(theta), minor * math.sin(theta)], dtype=np.float32)
        points.append(state.pos + rotate(local, state.heading))
    return points


class Renderer:
    """Interactive pygame renderer.

    Importing this module is headless-safe. Pygame is imported and initialized only
    when callers explicitly construct ``Renderer``.
    """

    def __init__(
        self,
        world_size: np.ndarray,
        window_size: tuple[int, int] = (1200, 800),
        fps: int = 45,
        trail_length: int = 90,
        meters_per_pixel: str | float = "auto",
    ) -> None:
        pygame = importlib.import_module("pygame")
        pygame.init()
        self.pygame = pygame
        try:
            self.gfxdraw = importlib.import_module("pygame.gfxdraw")
        except ImportError:
            self.gfxdraw = None

        self.world_size = np.asarray(world_size, dtype=np.float32)
        if self.world_size.shape != (2,) or not np.all(np.isfinite(self.world_size)):
            raise ValueError("world_size must be a finite 2-vector")
        if np.any(self.world_size <= 0.0):
            raise ValueError("world_size must be positive")

        self.window_size = tuple(int(value) for value in window_size)
        self.fps = int(fps)
        self.trail_length = int(trail_length)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Dynamic predictive encirclement")
        self.clock = pygame.time.Clock()
        self.axis_font = pygame.font.Font(None, 18)
        self.axis_title_font = pygame.font.Font(None, 20)
        self.status_font = pygame.font.Font(None, 22)
        self._closed = False

        if str(meters_per_pixel) == "auto":
            self.scale = min(
                self.window_size[0] / self.world_size[0],
                self.window_size[1] / self.world_size[1],
            ) * 0.92
        else:
            meters = float(meters_per_pixel)
            if not math.isfinite(meters) or meters <= 0.0:
                raise ValueError("meters_per_pixel must be positive or 'auto'")
            self.scale = 1.0 / meters
        self.offset = np.array(
            [
                (self.window_size[0] - self.world_size[0] * self.scale) * 0.5,
                (self.window_size[1] - self.world_size[1] * self.scale) * 0.5,
            ],
            dtype=np.float32,
        )
        self.trails: list[deque[np.ndarray]] = []
        self.shadow_trail: deque[np.ndarray] = deque(maxlen=self.trail_length)

    def reset_trails(self) -> None:
        self.trails = []
        self.shadow_trail.clear()

    def pump_events(self) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                return False
        return True

    def world_to_screen(self, pos: np.ndarray) -> tuple[int, int]:
        value = np.asarray(pos, dtype=np.float32)
        x = self.offset[0] + value[0] * self.scale
        y = self.offset[1] + (self.world_size[1] - value[1]) * self.scale
        return int(round(float(x))), int(round(float(y)))

    def render(self, frame: RenderFrame) -> None:
        if self._closed:
            raise RuntimeError("cannot render after Renderer.close()")

        pygame = self.pygame
        surface_mask = self._active_mask(frame.surface_active_mask, len(frame.blue_states))
        submarine_mask = self._active_mask(
            frame.submarine_active_mask, len(frame.submarine_states)
        )
        states = [*frame.blue_states, *frame.submarine_states, frame.red_state]
        if len(self.trails) != len(states):
            self.trails = [deque(maxlen=self.trail_length) for _ in states]
        for trail, state in zip(self.trails, states):
            trail.append(np.asarray(state.pos, dtype=np.float32).copy())
        if frame.show_shadow_red and frame.shadow_red_state is not None:
            self.shadow_trail.append(
                np.asarray(frame.shadow_red_state.pos, dtype=np.float32).copy()
            )

        self.screen.fill(WORLD_BACKGROUND_FILL)
        world_rect = pygame.Rect(
            int(self.offset[0]),
            int(self.offset[1]),
            int(frame.world_size[0] * self.scale),
            int(frame.world_size[1] * self.scale),
        )
        pygame.draw.rect(self.screen, WORLD_BACKGROUND_FILL, world_rect)
        self._draw_grid_and_axes(frame, world_rect)
        self._draw_island(frame, world_rect)
        pygame.draw.rect(self.screen, AXIS_LINE_COLOR, world_rect, width=2)

        submarine_color = submarine_fill_color(
            frame.submarine_states[0],
            frame.cooperative_task_success,
        ) if frame.submarine_states else SUBMERGED_SUBMARINE_FILL
        surface_trail_colors = [BLUE_SHIP_FILL, BLUE_SHIP_2_FILL]
        trail_colors = (
            [surface_trail_colors[index % 2] for index in range(len(frame.blue_states))]
            + [submarine_color] * len(frame.submarine_states)
            + [RED_SHIP_FILL]
        )
        visible = [*surface_mask, *submarine_mask, True]
        for trail, color, is_visible in zip(self.trails, trail_colors, visible):
            if not is_visible:
                continue
            if len(trail) >= 2:
                pygame.draw.lines(
                    self.screen,
                    color,
                    False,
                    [self.world_to_screen(position) for position in trail],
                    2,
                )
        if frame.show_shadow_red and len(self.shadow_trail) >= 2:
            self._draw_dashed_polyline(
                [self.world_to_screen(position) for position in self.shadow_trail],
                SHADOW_RED_COLOR,
                width=2,
            )

        self._draw_route(frame, world_rect)
        if frame.encirclement_valid and len(frame.submarine_states)==3:
            # Draw the actual enclosing hull only when the geometric gates
            # hold. These are vessel positions, never prescribed vertices.
            positions=[state.pos for state in frame.submarine_states]
            positions.sort(key=lambda p: math.atan2(p[1]-frame.red_state.pos[1], p[0]-frame.red_state.pos[0]))
            pygame.draw.lines(self.screen,SUCCESS_SUBMARINE_FILL,True,
                              [self.world_to_screen(p) for p in positions],1)
        surface_fills = [BLUE_SHIP_FILL, BLUE_SHIP_2_FILL]
        surface_outlines = [BLUE_SHIP_OUTLINE, BLUE_SHIP_2_OUTLINE]
        for index, (state, is_visible) in enumerate(
            zip(frame.blue_states, surface_mask), start=1
        ):
            if not is_visible:
                continue
            self._draw_ship(
                state,
                surface_fills[(index - 1) % 2],
                surface_outlines[(index - 1) % 2],
                frame.semi_major,
                frame.semi_minor,
            )
            self._draw_entity_label(
                state, f"B{index}", surface_outlines[(index - 1) % 2]
            )
        for index, (state, is_visible) in enumerate(
            zip(frame.submarine_states, submarine_mask), start=1
        ):
            if not is_visible:
                continue
            self._draw_submarine(
                state,
                frame.semi_major,
                frame.semi_minor,
                cooperative_task_success=frame.cooperative_task_success,
            )
            self._draw_entity_label(state, f"Y{index}", SUBMARINE_OUTLINE)
        self._draw_ship(
            frame.red_state,
            RED_SHIP_FILL,
            RED_SHIP_OUTLINE,
            frame.semi_major,
            frame.semi_minor,
        )
        self._draw_status(frame, world_rect)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.pygame.display.quit()
        self.pygame.quit()

    def _length(self, meters: float) -> int:
        return max(1, int(round(float(meters) * self.scale)))

    @staticmethod
    def _active_mask(value: np.ndarray | None, length: int) -> list[bool]:
        if value is None:
            return [True] * int(length)
        array = np.asarray(value).reshape(-1)
        if array.size != int(length):
            raise ValueError(f"active mask must contain {length} entries")
        if not np.all(np.isfinite(array)):
            raise ValueError("active mask must be finite")
        return [bool(item > 0.5) for item in array]

    def _draw_island(self, frame: RenderFrame, world_rect: Any) -> None:
        if frame.reefs is not None:
            previous_clip = self.screen.get_clip()
            self.screen.set_clip(world_rect)
            for reef in frame.reefs:
                center = self.world_to_screen(np.asarray(reef["center"]))
                radius = self._length(reef["radius"])
                self.pygame.draw.circle(self.screen, ISLAND_FILL, center, radius)
                self.pygame.draw.circle(self.screen, ISLAND_OUTLINE, center, radius, width=2)
                label = self.axis_font.render(str(reef.get("name", "")), True, ISLAND_OUTLINE)
                self.screen.blit(label, label.get_rect(center=center))
            if frame.protected_center is not None:
                center = self.world_to_screen(frame.protected_center)
                radius = self._length(frame.protected_radius)
                self.pygame.draw.circle(self.screen, (220, 237, 255), center, radius)
                self.pygame.draw.circle(self.screen, BLUE_SHIP_OUTLINE, center, radius, width=2)
            self.screen.set_clip(previous_clip)
            return
        if not bool(getattr(frame, "island_active", True)) or frame.island_radius <= 0.0:
            return
        center = self.world_to_screen(frame.island_center)
        radius = self._length(frame.island_radius)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(world_rect)
        self.pygame.draw.circle(self.screen, ISLAND_FILL, center, radius)
        self.pygame.draw.circle(self.screen, ISLAND_OUTLINE, center, radius, width=2)
        self.screen.set_clip(previous_clip)

    def _draw_route(self, frame: RenderFrame, world_rect: Any) -> None:
        # Reef interception has no prescribed route. Its actual APF trail is
        # already drawn above; a straight goal line cuts through the reefs
        # and can be mistaken for the vessel's commanded or executed path.
        if frame.task_name == "reef interception":
            return
        surface_mask = self._active_mask(
            frame.surface_active_mask, len(frame.blue_states)
        )
        if not any(surface_mask):
            return
        start = np.asarray(frame.route_start, dtype=np.float32)
        end = np.asarray(frame.route_end, dtype=np.float32)
        route = end - start
        route_length = float(np.linalg.norm(route))
        if route_length <= 1e-6:
            return
        direction = route / route_length
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(world_rect)
        dash_period = 20.0 * float(np.min(frame.world_size)) / 55.0
        for distance in np.arange(0.0, route_length, dash_period):
            dash_start = start + direction * distance
            dash_end = start + direction * min(distance + 0.5 * dash_period, route_length)
            self.pygame.draw.line(
                self.screen,
                ROUTE_COLOR,
                self.world_to_screen(dash_start),
                self.world_to_screen(dash_end),
                2,
            )
        self.screen.set_clip(previous_clip)


    def _draw_cross(self, center: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
        self.pygame.draw.line(
            self.screen,
            color,
            (center[0] - radius, center[1]),
            (center[0] + radius, center[1]),
            width=2,
        )
        self.pygame.draw.line(
            self.screen,
            color,
            (center[0], center[1] - radius),
            (center[0], center[1] + radius),
            width=2,
        )

    def _draw_entity_label(
        self,
        state: ShipState,
        label_text: str,
        color: tuple[int, int, int],
    ) -> None:
        point = self.world_to_screen(state.pos)
        label = self.axis_font.render(label_text, True, color)
        self.screen.blit(label, label.get_rect(midbottom=(point[0], point[1] - 8)))

    def _draw_status(self, frame: RenderFrame, world_rect: Any) -> None:
        phase = getattr(frame.phase, "value", frame.phase)
        stage = str(getattr(frame, "task_name", "final_task") or "final_task")
        band = str(getattr(frame, "difficulty_band", "") or "")
        stage_text = stage if not band else f"{stage} [{band}]"
        prefix = f"{stage_text}   " if stage_text else ""
        status = f"{prefix}{phase}"
        label = self.status_font.render(status, True, AXIS_TEXT_COLOR)
        self.screen.blit(label, label.get_rect(topleft=(world_rect.left + 8, world_rect.top + 8)))
        details: list[str] = []
        metadata = getattr(frame, "visualization_metadata", None) or {}
        if metadata:
            checkpoint = str(
                metadata.get(
                    "checkpoint_path",
                    metadata.get("checkpoint", metadata.get("checkpoint_type", "")),
                )
            )
            if checkpoint:
                details.append(f"checkpoint: {checkpoint}")
            identity_parts = []
            for key, label_name in (
                ("experiment", "experiment"),
                ("simulation_step", "step"),
                ("checkpoint_run_id", "run"),
                ("seed", "seed"),
                ("training_episode", "episode"),
                ("global_env_steps", "env_steps"),
            ):
                if key in metadata and metadata[key] is not None:
                    identity_parts.append(f"{label_name}={metadata[key]}")
            if identity_parts:
                details.append("  ".join(identity_parts))
            evaluation_parts = []
            for key, label_name in (
                ("fixed_eval_success_rate", "fixed_success"),
                ("random_eval_success_rate", "random_success"),
            ):
                value = metadata.get(key)
                if value is not None:
                    evaluation_parts.append(f"{label_name}={float(value):.3f}")
            if evaluation_parts:
                details.append("  ".join(evaluation_parts))
            sources = metadata.get("actor_source_checkpoint")
            if sources:
                details.append(f"actor_source: {sources}")
        details.append(f"t={float(getattr(frame, 'simulation_time', 0.0)):.2f}s")
        if frame.red_behavior:
            details.append('red: ' + frame.red_behavior)
        events = getattr(frame, "event_markers", None) or {}
        active_events = [name for name, active in events.items() if bool(active)]
        if active_events:
            details.append("events: " + ", ".join(active_events))
        reason = str(getattr(frame, "termination_reason", "") or "")
        if reason:
            details.append("termination: " + reason)
        diagnostics = getattr(frame, "diagnostics", None) or {}
        if diagnostics:
            ordered = ("radius_ref", "hold_progress", "capture_max_m", "enclosure_gap_deg", "e_cf", "clearance_gain", "withdrawal_remaining_m")
            values = [
                f"{key}={float(diagnostics[key]):.2f}"
                for key in ordered
                if key in diagnostics and np.isfinite(float(diagnostics[key]))
            ]
            if values:
                details.append("  ".join(values))
        for row, detail in enumerate(details, start=1):
            detail_label = self.axis_font.render(detail, True, AXIS_TEXT_COLOR)
            self.screen.blit(
                detail_label,
                detail_label.get_rect(
                    topleft=(world_rect.left + 8, world_rect.top + 8 + row * 19)
                ),
            )
        if frame.cooperative_task_success:
            success = self.status_font.render("COOPERATIVE SUCCESS", True, SUCCESS_SUBMARINE_FILL)
            self.screen.blit(success, success.get_rect(topright=(world_rect.right - 8, world_rect.top + 8)))
        elif frame.physical_capture_success:
            physical = self.status_font.render("PHYSICAL CAPTURE", True, PHYSICAL_READY_COLOR)
            self.screen.blit(physical, physical.get_rect(topright=(world_rect.right - 8, world_rect.top + 8)))

    def _draw_dashed_polyline(
        self,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        for index in range(0, len(points) - 1, 2):
            self.pygame.draw.line(
                self.screen, color, points[index], points[index + 1], width
            )

    def _draw_grid_and_axes(self, frame: RenderFrame, world_rect: Any) -> None:
        grid_step = 1.0 if float(np.max(frame.world_size)) <= 15.0 else GRID_STEP_METERS
        x_ticks = axis_tick_values(float(frame.world_size[0]), grid_step)
        y_ticks = axis_tick_values(float(frame.world_size[1]), grid_step)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(world_rect)
        for x_value in x_ticks:
            start = self.world_to_screen(np.array([x_value, 0.0], dtype=np.float32))
            end = self.world_to_screen(
                np.array([x_value, float(frame.world_size[1])], dtype=np.float32)
            )
            self.pygame.draw.line(self.screen, GRID_LINE_COLOR, start, end, width=1)
        for y_value in y_ticks:
            start = self.world_to_screen(np.array([0.0, y_value], dtype=np.float32))
            end = self.world_to_screen(
                np.array([float(frame.world_size[0]), y_value], dtype=np.float32)
            )
            self.pygame.draw.line(self.screen, GRID_LINE_COLOR, start, end, width=1)
        self.screen.set_clip(previous_clip)

        for x_value in x_ticks:
            screen_x, _ = self.world_to_screen(np.array([x_value, 0.0], dtype=np.float32))
            label = self.axis_font.render(f"{x_value:.0f}", True, AXIS_TEXT_COLOR)
            self.screen.blit(label, label.get_rect(midtop=(screen_x, world_rect.bottom + 3)))
        for y_value in y_ticks:
            _, screen_y = self.world_to_screen(np.array([0.0, y_value], dtype=np.float32))
            label = self.axis_font.render(f"{y_value:.0f}", True, AXIS_TEXT_COLOR)
            self.screen.blit(label, label.get_rect(midright=(world_rect.left - 6, screen_y)))

        x_title = self.axis_title_font.render("X (m)", True, AXIS_TEXT_COLOR)
        self.screen.blit(x_title, x_title.get_rect(midtop=(world_rect.centerx, world_rect.bottom + 17)))
        y_title = self.pygame.transform.rotate(
            self.axis_title_font.render("Y (m)", True, AXIS_TEXT_COLOR),
            90,
        )
        self.screen.blit(y_title, y_title.get_rect(midleft=(4, world_rect.centery)))

    def _draw_ship(
        self,
        state: ShipState,
        fill: tuple[int, int, int],
        outline: tuple[int, int, int],
        semi_major: float,
        semi_minor: float,
    ) -> None:
        points = self._ship_polygon(state, semi_major, semi_minor)
        self._draw_smooth_polygon(points, fill, outline, width=2)
        nose_vector = (
            np.array([math.cos(state.heading), math.sin(state.heading)], dtype=np.float32)
            * float(semi_major)
            * 1.15
        )
        nose = self.world_to_screen(state.pos + nose_vector)
        center = self.world_to_screen(state.pos)
        self.pygame.draw.aaline(self.screen, outline, center, nose)

    def _draw_submarine(
        self,
        state: ShipState,
        semi_major: float,
        semi_minor: float,
        cooperative_task_success: bool,
    ) -> None:
        points = [
            self.world_to_screen(point)
            for point in submarine_world_ellipse_points(state, semi_major, semi_minor)
        ]
        self._draw_smooth_polygon(
            points,
            submarine_fill_color(
                state,
                cooperative_task_success,
            ),
            SUBMARINE_OUTLINE,
            width=2,
        )
        nose_vector = (
            np.array([math.cos(state.heading), math.sin(state.heading)], dtype=np.float32)
            * float(semi_major)
        )
        nose = self.world_to_screen(state.pos + nose_vector)
        center = self.world_to_screen(state.pos)
        self.pygame.draw.aaline(self.screen, SUBMARINE_OUTLINE, center, nose)

    def _draw_smooth_polygon(
        self,
        points: list[tuple[int, int]],
        fill: tuple[int, int, int],
        outline: tuple[int, int, int],
        width: int,
    ) -> None:
        if self.gfxdraw is not None:
            try:
                self.gfxdraw.filled_polygon(self.screen, points, fill)
                self.gfxdraw.aapolygon(self.screen, points, outline)
                self.pygame.draw.aalines(self.screen, outline, True, points)
                return
            except (TypeError, ValueError):
                pass
        self.pygame.draw.polygon(self.screen, fill, points)
        self.pygame.draw.polygon(self.screen, outline, points, width=width)

    def _ship_polygon(
        self,
        state: ShipState,
        semi_major: float,
        semi_minor: float,
    ) -> list[tuple[int, int]]:
        major = float(semi_major)
        minor = float(semi_minor)
        left_cut = -0.75 * major
        right_cut = 0.75 * major
        points: list[np.ndarray] = []
        left_theta = math.acos(left_cut / major)
        right_theta = math.acos(right_cut / major)
        for theta in np.linspace(left_theta, right_theta, 14):
            points.append(
                np.array([major * math.cos(theta), minor * math.sin(theta)], dtype=np.float32)
            )
        points.append(np.array([major * 1.28, 0.0], dtype=np.float32))
        for theta in np.linspace(-right_theta, -left_theta, 14):
            points.append(
                np.array([major * math.cos(theta), minor * math.sin(theta)], dtype=np.float32)
            )
        world_points = [state.pos + rotate(point, state.heading) for point in points]
        return [self.world_to_screen(point) for point in world_points]
