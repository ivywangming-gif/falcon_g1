"""Search-first access-aware planning primitives for physical box pushing.

The module is deliberately independent of Isaac Lab.  Isaac is the authority
for the full-G1 replay certificate; this module only admits that certificate
as an input and never converts an unchecked geometric path into a valid edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Callable, Iterable, Sequence

import numpy as np


FACE_NAMES = ("rear", "front", "right", "left")
FACE_LAMBDA = {"rear": 0, "front": 1, "right": 2, "left": 3}
LAMBDA_FACE = {value: key for key, value in FACE_LAMBDA.items()}


def wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class BoxPose:
    x: float
    y: float
    yaw: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.asarray((self.x, self.y), dtype=np.float64)


@dataclass(frozen=True)
class BoxGeometry:
    length_m: float = 1.40
    width_m: float = 0.70
    height_m: float = 0.80


@dataclass(frozen=True)
class Obstacle:
    name: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        cx, cy = self.center_xy
        sx, sy = self.size_xy
        return cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0


@dataclass(frozen=True)
class StanceTemplate:
    face: str
    pose: BoxPose
    normal_gap_m: float
    source: str
    tangent_offset_m: float = 0.0


@dataclass(frozen=True)
class PushPrimitive:
    primitive_id: str
    source_face: str
    length_m: float
    curvature_m_inv: float = 0.0
    samples: int = 21
    direction_local: tuple[float, float] = (1.0, 0.0)
    family: str | None = None

    @property
    def family_id(self) -> str:
        return self.primitive_id if self.family is None else self.family


@dataclass(frozen=True)
class PlannerState:
    """Markov state ``(q_box, lambda, previous_primitive)``.

    ``interaction_lambda`` identifies the occupied interaction template.  A
    PUSH changes q and keeps lambda; a REPOSITION keeps q and changes lambda.
    """

    q_box: BoxPose
    interaction_lambda: int
    previous_primitive: str | None

    @property
    def face(self) -> str:
        try:
            return LAMBDA_FACE[int(self.interaction_lambda)]
        except KeyError as exc:
            raise ValueError(f"invalid interaction lambda: {self.interaction_lambda}") from exc


@dataclass(frozen=True)
class RepositionCertificate:
    passed: bool
    max_tracking_error_m: float | None = None
    min_clearance_m: float | None = None
    source: str = ""
    reason: str = ""


@dataclass
class PlannerEdge:
    edge_id: str
    edge_type: str
    source: PlannerState | None
    target: PlannerState | None
    source_face: str | None = None
    target_face: str | None = None
    box_poses: list[BoxPose] = field(default_factory=list)
    base_path_xy: list[tuple[float, float]] = field(default_factory=list)
    checks: dict[str, object] = field(default_factory=dict)
    cost: float = 0.0


@dataclass(frozen=True)
class AStarResult:
    passed: bool
    path_xy: tuple[tuple[float, float], ...] = ()
    length_m: float = 0.0
    expanded_nodes: int = 0
    reason: str = ""


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s), (s, c)), dtype=np.float64)


def box_corners(pose: BoxPose, geometry: BoxGeometry) -> np.ndarray:
    half = np.asarray((geometry.length_m / 2.0, geometry.width_m / 2.0), dtype=np.float64)
    local = np.asarray(((-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]), (-half[0], half[1])))
    return pose.xy[None, :] + (_rotation(pose.yaw) @ local.T).T


def polygons_overlap(first: np.ndarray, second: np.ndarray, *, touch_is_collision: bool = True) -> bool:
    """SAT overlap for convex quadrilaterals; touching is conservatively bad."""
    axes: list[np.ndarray] = []
    for polygon in (first, second):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            normal = np.asarray((-edge[1], edge[0]), dtype=np.float64)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                axes.append(normal / norm)
    for axis in axes:
        first_projection = first @ axis
        second_projection = second @ axis
        if touch_is_collision:
            separated = first_projection.max() < second_projection.min() - 1e-10 or second_projection.max() < first_projection.min() - 1e-10
        else:
            separated = first_projection.max() <= second_projection.min() + 1e-10 or second_projection.max() <= first_projection.min() + 1e-10
        if separated:
            return False
    return True


def box_footprint_clear(pose: BoxPose, geometry: BoxGeometry, obstacles: Sequence[Obstacle]) -> tuple[bool, tuple[str, ...]]:
    footprint = box_corners(pose, geometry)
    names = []
    for obstacle in obstacles:
        bounds = obstacle.bounds
        rectangle = np.asarray(((bounds[0], bounds[2]), (bounds[1], bounds[2]), (bounds[1], bounds[3]), (bounds[0], bounds[3])), dtype=np.float64)
        if polygons_overlap(footprint, rectangle):
            names.append(obstacle.name)
    return not names, tuple(names)


def sample_box_poses(start: BoxPose, primitive: PushPrimitive) -> list[BoxPose]:
    count = max(2, int(primitive.samples))
    direction = np.asarray(primitive.direction_local, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction.shape != (2,) or direction_norm <= 1e-12:
        raise ValueError("direction_local must be a non-zero 2-vector")
    direction /= direction_norm
    heading = math.atan2(float(direction[1]), float(direction[0]))
    values = []
    for fraction in np.linspace(0.0, 1.0, count):
        s = float(primitive.length_m) * float(fraction)
        if abs(primitive.curvature_m_inv) <= 1e-12:
            local = direction * s
        else:
            kappa = float(primitive.curvature_m_inv)
            local_curve = np.asarray((math.sin(kappa * s) / kappa, (1.0 - math.cos(kappa * s)) / kappa), dtype=np.float64)
            local = _rotation(heading) @ local_curve
        world = start.xy + _rotation(start.yaw) @ local
        values.append(BoxPose(float(world[0]), float(world[1]), wrap_angle(start.yaw + primitive.curvature_m_inv * s)))
    return values


def stance_template(
    box: BoxPose,
    face: str,
    normal_gap_m: float = 0.02,
    geometry: BoxGeometry | None = None,
    *,
    tangent_offset_m: float = 0.0,
    relative_yaw_rad: float | None = None,
    source: str = "dimension-derived-face-template",
) -> StanceTemplate:
    if face not in FACE_NAMES:
        raise ValueError(f"unknown face: {face}")
    if geometry is None:
        geometry = BoxGeometry()
    half_length = geometry.length_m / 2.0
    half_width = geometry.width_m / 2.0
    normal = {
        "rear": np.asarray((-1.0, 0.0)),
        "front": np.asarray((1.0, 0.0)),
        "right": np.asarray((0.0, -1.0)),
        "left": np.asarray((0.0, 1.0)),
    }[face]
    tangent = {
        "rear": np.asarray((0.0, 1.0)),
        "front": np.asarray((0.0, -1.0)),
        "right": np.asarray((1.0, 0.0)),
        "left": np.asarray((-1.0, 0.0)),
    }[face]
    half_extent = half_length if face in ("rear", "front") else half_width
    local_xy = normal * (half_extent + float(normal_gap_m)) + tangent * float(tangent_offset_m)
    default_yaw = {
        "rear": 0.0,
        "front": math.pi,
        "right": math.pi / 2.0,
        "left": -math.pi / 2.0,
    }[face]
    relative_yaw = default_yaw if relative_yaw_rad is None else float(relative_yaw_rad)
    xy = box.xy + _rotation(box.yaw) @ local_xy
    return StanceTemplate(
        face,
        BoxPose(float(xy[0]), float(xy[1]), wrap_angle(box.yaw + relative_yaw)),
        float(normal_gap_m),
        source,
        float(tangent_offset_m),
    )


def _point_in_inflated_obstacle(point: np.ndarray, obstacle: Obstacle, radius_m: float) -> bool:
    xmin, xmax, ymin, ymax = obstacle.bounds
    dx = max(xmin - float(point[0]), 0.0, float(point[0]) - xmax)
    dy = max(ymin - float(point[1]), 0.0, float(point[1]) - ymax)
    return math.hypot(dx, dy) <= radius_m + 1e-12


def _point_segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    delta = second - first
    denom = float(delta @ delta)
    if denom <= 1e-16:
        return float(np.linalg.norm(point - first))
    fraction = min(1.0, max(0.0, float((point - first) @ delta) / denom))
    return float(np.linalg.norm(point - (first + fraction * delta)))


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    signs = []
    for index in range(len(polygon)):
        edge = polygon[(index + 1) % len(polygon)] - polygon[index]
        rel = point - polygon[index]
        signs.append(float(edge[0] * rel[1] - edge[1] * rel[0]))
    return all(value >= -1e-12 for value in signs) or all(value <= 1e-12 for value in signs)


def _point_in_inflated_polygon(point: np.ndarray, polygon: np.ndarray, radius_m: float) -> bool:
    if _point_in_convex_polygon(point, polygon):
        return True
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ) <= radius_m + 1e-12


def astar_base_path(
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    box: BoxPose,
    geometry: BoxGeometry,
    obstacles: Sequence[Obstacle],
    robot_radius_m: float,
    tracking_margin_m: float,
    resolution_m: float = 0.05,
    bounds: tuple[float, float, float, float] | None = None,
    *,
    weight: float = 1.0,
) -> AStarResult:
    """Weighted 8-connected A* inflated by the physical tracking tube."""
    if resolution_m <= 0.0 or robot_radius_m < 0.0 or tracking_margin_m < 0.0 or weight < 1.0:
        return AStarResult(False, reason="INVALID_ASTAR_PARAMETERS")
    start = np.asarray(start_xy, dtype=np.float64)
    goal = np.asarray(goal_xy, dtype=np.float64)
    if start.shape != (2,) or goal.shape != (2,) or not np.isfinite(start).all() or not np.isfinite(goal).all():
        return AStarResult(False, reason="NONFINITE_ENDPOINT")
    inflation = float(robot_radius_m + tracking_margin_m)
    if bounds is None:
        box_radius = math.hypot(geometry.length_m / 2.0, geometry.width_m / 2.0) + inflation
        candidates = np.vstack((start, goal, box.xy - box_radius, box.xy + box_radius))
        # Robot reposition is local around a fixed box. Distant obstacles do
        # not enlarge the grid, although ``blocked`` still checks all of them.
        low = candidates.min(axis=0) - 0.75
        high = candidates.max(axis=0) + 0.75
    else:
        xmin, ymin, xmax, ymax = bounds
        low = np.asarray((xmin, ymin), dtype=np.float64)
        high = np.asarray((xmax, ymax), dtype=np.float64)
    origin = np.floor(low / resolution_m) * resolution_m
    shape = np.maximum(1, np.ceil((high - origin) / resolution_m).astype(int) + 1)

    def node(point: np.ndarray) -> tuple[int, int]:
        return tuple(np.rint((point - origin) / resolution_m).astype(int))

    def point(cell: tuple[int, int]) -> np.ndarray:
        return origin + resolution_m * np.asarray(cell, dtype=np.float64)

    source, target = node(start), node(goal)

    def blocked(cell: tuple[int, int]) -> bool:
        if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
            return True
        current = point(cell)
        local = _rotation(-box.yaw) @ (current - box.xy)
        dx = max(abs(float(local[0])) - geometry.length_m / 2.0, 0.0)
        dy = max(abs(float(local[1])) - geometry.width_m / 2.0, 0.0)
        if math.hypot(dx, dy) <= inflation + 1e-12:
            return True
        return any(_point_in_inflated_obstacle(current, obstacle, inflation) for obstacle in obstacles)

    if blocked(source) or blocked(target):
        return AStarResult(False, reason="ASTAR_ENDPOINT_BLOCKED")
    frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, source)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    costs = {source: 0.0}
    expanded = 0
    moves = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost > costs.get(current, math.inf) + 1e-12:
            continue
        expanded += 1
        if current == target:
            break
        for dx, dy in moves:
            nxt = (current[0] + dx, current[1] + dy)
            if blocked(nxt):
                continue
            if dx and dy and (blocked((current[0] + dx, current[1])) or blocked((current[0], current[1] + dy))):
                continue
            new_cost = current_cost + (math.sqrt(2.0) if dx and dy else 1.0)
            if new_cost < costs.get(nxt, math.inf):
                costs[nxt] = new_cost
                came[nxt] = current
                heuristic = math.hypot(target[0] - nxt[0], target[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + weight * heuristic, new_cost, nxt))
    if target != source and target not in came:
        return AStarResult(False, expanded_nodes=expanded, reason="ASTAR_NO_CONNECTIVITY")
    cells = [target]
    while cells[-1] != source:
        cells.append(came[cells[-1]])
    cells.reverse()
    path = tuple(tuple(float(value) for value in point(cell)) for cell in cells)
    length = float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(path[:-1], path[1:])))
    return AStarResult(True, path, length, expanded, "")


class AccessAwarePlanner:
    """State-space search with explicit geometry and full-G1 edge gates."""

    def __init__(
        self,
        geometry: BoxGeometry,
        obstacles: Sequence[Obstacle],
        *,
        robot_radius_m: float,
        tracking_margin_m: float,
        stance_validator: Callable[[BoxPose, StanceTemplate], bool],
        full_g1_replay: Callable[[BoxPose, StanceTemplate, Sequence[tuple[float, float]]], RepositionCertificate],
        initial_base_xy: Sequence[float] | None = None,
        normal_gap_m: float = 0.02,
        normal_gap_by_face: dict[str, float] | None = None,
        tangent_offset_by_face: dict[str, float] | None = None,
        relative_yaw_by_face: dict[str, float] | None = None,
        template_source: str = "dimension-derived-face-template",
        astar_resolution_m: float = 0.05,
        astar_weight: float = 1.25,
        astar_bounds: tuple[float, float, float, float] | None = None,
        reposition_fixed_cost_m: float = 0.20,
        reposition_path_weight: float = 1.0,
        primitive_switch_cost_m: float = 0.10,
        turn_cost_weight_m_per_rad: float = 0.20,
        state_xy_resolution_m: float = 0.01,
        state_yaw_resolution_rad: float = 0.01,
    ) -> None:
        self.geometry = geometry
        self.obstacles = tuple(obstacles)
        self.robot_radius_m = float(robot_radius_m)
        self.tracking_margin_m = float(tracking_margin_m)
        self.stance_validator = stance_validator
        self.full_g1_replay = full_g1_replay
        self.initial_base_xy = None if initial_base_xy is None else np.asarray(initial_base_xy, dtype=np.float64)
        self.normal_gap_m = float(normal_gap_m)
        self.normal_gap_by_face = dict(normal_gap_by_face or {})
        self.tangent_offset_by_face = dict(tangent_offset_by_face or {})
        self.relative_yaw_by_face = dict(relative_yaw_by_face or {})
        self.template_source = str(template_source)
        self.astar_resolution_m = float(astar_resolution_m)
        self.astar_weight = float(astar_weight)
        self.astar_bounds = astar_bounds
        self.reposition_fixed_cost_m = float(reposition_fixed_cost_m)
        self.reposition_path_weight = float(reposition_path_weight)
        self.primitive_switch_cost_m = float(primitive_switch_cost_m)
        self.turn_cost_weight_m_per_rad = float(turn_cost_weight_m_per_rad)
        self.state_xy_resolution_m = float(state_xy_resolution_m)
        self.state_yaw_resolution_rad = float(state_yaw_resolution_rad)
        self._reposition_cache: dict[
            tuple[object, ...], tuple[AStarResult, RepositionCertificate, BoxPose]
        ] = {}
        self.reposition_cache_hits = 0
        self.reposition_cache_misses = 0
        self._forbidden_lazy_repositions: set[tuple[int, int, int, int, int]] = set()

    def _stance(self, box: BoxPose, face: str, *, clearance: bool = False) -> StanceTemplate:
        if clearance:
            gap = self.robot_radius_m + self.tracking_margin_m + self.astar_resolution_m
            tangent = 0.0
            source = "planner-clearance-endpoint"
        else:
            gap = float(self.normal_gap_by_face.get(face, self.normal_gap_m))
            tangent = float(self.tangent_offset_by_face.get(face, 0.0))
            source = self.template_source
        return stance_template(
            box,
            face,
            gap,
            self.geometry,
            tangent_offset_m=tangent,
            relative_yaw_rad=self.relative_yaw_by_face.get(face),
            source=source,
        )

    def _initial_stance_ok(self, box: BoxPose, face: str) -> tuple[bool, StanceTemplate, AStarResult | None]:
        stance = self._stance(box, face)
        if not self.stance_validator(box, stance):
            return False, stance, None
        if self.initial_base_xy is None:
            return True, stance, None
        clearance_stance = self._stance(box, face, clearance=True)
        base_path = astar_base_path(
            self.initial_base_xy, clearance_stance.pose.xy, box, self.geometry, self.obstacles,
            self.robot_radius_m, self.tracking_margin_m, self.astar_resolution_m, self.astar_bounds,
            weight=self.astar_weight,
        )
        return base_path.passed, stance, base_path

    def _push_cost(self, state: PlannerState, primitive: PushPrimitive) -> tuple[float, dict[str, float]]:
        switch_cost = self.primitive_switch_cost_m if (
            state.previous_primitive is not None and state.previous_primitive != primitive.family_id
        ) else 0.0
        turn_cost = self.turn_cost_weight_m_per_rad * abs(primitive.curvature_m_inv * primitive.length_m)
        return float(primitive.length_m + switch_cost + turn_cost), {
            "spatial_base_cost_m": float(primitive.length_m),
            "primitive_switch_cost_m": float(switch_cost),
            "turn_cost_m": float(turn_cost),
        }

    def push_successor(
        self,
        state: PlannerState,
        primitive: PushPrimitive,
        *,
        terminal_goal_connector: bool = False,
    ) -> PlannerEdge | None:
        if primitive.source_face != state.face:
            return None
        poses = sample_box_poses(state.q_box, primitive)
        collision_rows = [box_footprint_clear(pose, self.geometry, self.obstacles) for pose in poses]
        if not all(row[0] for row in collision_rows):
            return None
        # A push edge is valid only when the physical interaction stance is
        # valid along the complete sampled box trajectory, not merely at the
        # terminal pose.  The validator is the Isaac-backed physical stance
        # contract supplied by the caller.
        stance_rows = []
        for pose in poses:
            stance = self._stance(pose, primitive.source_face)
            valid = bool(self.stance_validator(pose, stance))
            stance_rows.append((valid, stance))
            if not valid:
                return None
        cost, cost_components = self._push_cost(state, primitive)
        target = PlannerState(poses[-1], state.interaction_lambda, primitive.family_id)
        return PlannerEdge(
            edge_id=primitive.primitive_id, edge_type="PUSH", source=state, target=target,
            source_face=primitive.source_face, target_face=primitive.source_face,
            box_poses=poses, checks={
                "box_footprint_collision_pass": True,
                "stance_validity_pass": True,
                "stance_sample_count": len(stance_rows),
                "complete_intermediate_path_checked": True,
                "lambda_unchanged": target.interaction_lambda == state.interaction_lambda,
                "terminal_goal_connector": bool(terminal_goal_connector),
                "L_m": float(primitive.length_m),
                "kappa_m_inv": float(primitive.curvature_m_inv),
                "direction_local": list(primitive.direction_local),
                "q_start": state.q_box,
                "q_ref": poses,
                "q_goal": poses[-1],
                "face": primitive.source_face,
                "cost_components": cost_components,
            },
            cost=cost,
        )

    def reposition_successor(self, state: PlannerState, target_face: str) -> PlannerEdge | None:
        if target_face not in FACE_NAMES or target_face == state.face:
            return None
        source_stance = self._stance(state.q_box, state.face)
        target_stance = self._stance(state.q_box, target_face)
        if not self.stance_validator(state.q_box, source_stance) or not self.stance_validator(state.q_box, target_stance):
            return None
        translation_invariant = not self.obstacles
        key = (
            "FREE_SPACE" if translation_invariant else "SCENE",
            0.0 if translation_invariant else round(state.q_box.x, 4),
            0.0 if translation_invariant else round(state.q_box.y, 4),
            round(state.q_box.yaw, 4), int(state.interaction_lambda), FACE_LAMBDA[target_face],
        )
        if key in self._reposition_cache:
            self.reposition_cache_hits += 1
            cached_path, certificate, cached_box = self._reposition_cache[key]
            if translation_invariant:
                delta = state.q_box.xy - cached_box.xy
                translated = tuple(tuple((np.asarray(point) + delta).tolist()) for point in cached_path.path_xy)
                base_path = AStarResult(
                    cached_path.passed, translated, cached_path.length_m, cached_path.expanded_nodes, cached_path.reason,
                )
            else:
                base_path = cached_path
        else:
            self.reposition_cache_misses += 1
            source_clearance = self._stance(state.q_box, state.face, clearance=True)
            target_clearance = self._stance(state.q_box, target_face, clearance=True)
            base_path = astar_base_path(
                source_clearance.pose.xy, target_clearance.pose.xy, state.q_box, self.geometry, self.obstacles,
                self.robot_radius_m, self.tracking_margin_m, self.astar_resolution_m, self.astar_bounds,
                weight=self.astar_weight,
            )
            certificate = self.full_g1_replay(state.q_box, target_stance, base_path.path_xy) if base_path.passed else RepositionCertificate(
                False, source="A_STAR", reason=base_path.reason,
            )
            self._reposition_cache[key] = (base_path, certificate, state.q_box)
        if not base_path.passed:
            return None
        tracking_pass = certificate.max_tracking_error_m is None or certificate.max_tracking_error_m <= self.tracking_margin_m + 1e-12
        clearance_pass = certificate.min_clearance_m is None or certificate.min_clearance_m > 0.0
        if not certificate.passed or not tracking_pass or not clearance_pass:
            return None
        target = PlannerState(state.q_box, FACE_LAMBDA[target_face], state.previous_primitive)
        edge_cost = self.reposition_fixed_cost_m + self.reposition_path_weight * base_path.length_m
        return PlannerEdge(
            edge_id=f"reposition_{state.face}_to_{target_face}", edge_type="REPOSITION", source=state, target=target,
            source_face=state.face, target_face=target_face, base_path_xy=list(base_path.path_xy),
            checks={
                "astar_connectivity_pass": True,
                "box_q_exactly_fixed": target.q_box == state.q_box,
                "lambda_changed": target.interaction_lambda != state.interaction_lambda,
                "previous_primitive_preserved": target.previous_primitive == state.previous_primitive,
                "lazy_full_g1_replay_cache": True,
                "full_g1_replay_pass": True,
                "tracking_tube_pass": tracking_pass,
                "clearance_positive_pass": clearance_pass,
                "certificate": certificate,
                "cost_components": {
                    "lambda_R_m": self.reposition_fixed_cost_m,
                    "w_B": self.reposition_path_weight,
                    "base_path_length_m": base_path.length_m,
                },
            },
            cost=float(edge_cost),
        )

    def _lazy_reposition_identity(self, state: PlannerState, target_face: str) -> tuple[int, int, int, int, int]:
        return (
            int(round(state.q_box.x / self.state_xy_resolution_m)),
            int(round(state.q_box.y / self.state_xy_resolution_m)),
            int(round(wrap_angle(state.q_box.yaw) / self.state_yaw_resolution_rad)),
            int(state.interaction_lambda),
            FACE_LAMBDA[target_face],
        )

    def lazy_reposition_successor(self, state: PlannerState, target_face: str) -> PlannerEdge | None:
        """Create a lower-bound edge; A* and replay run only for a goal candidate."""
        if target_face not in FACE_NAMES or target_face == state.face:
            return None
        if self._lazy_reposition_identity(state, target_face) in self._forbidden_lazy_repositions:
            return None
        source_stance = self._stance(state.q_box, state.face)
        target_stance = self._stance(state.q_box, target_face)
        if not self.stance_validator(state.q_box, source_stance) or not self.stance_validator(state.q_box, target_stance):
            return None
        source_clearance = self._stance(state.q_box, state.face, clearance=True)
        target_clearance = self._stance(state.q_box, target_face, clearance=True)
        chord = float(np.linalg.norm(target_clearance.pose.xy - source_clearance.pose.xy))
        target = PlannerState(state.q_box, FACE_LAMBDA[target_face], state.previous_primitive)
        return PlannerEdge(
            edge_id=f"reposition_{state.face}_to_{target_face}",
            edge_type="REPOSITION",
            source=state,
            target=target,
            source_face=state.face,
            target_face=target_face,
            checks={
                "box_q_exactly_fixed": target.q_box == state.q_box,
                "lambda_changed": target.interaction_lambda != state.interaction_lambda,
                "previous_primitive_preserved": target.previous_primitive == state.previous_primitive,
                "lazy_candidate_pending": True,
                "reposition_cost_lower_bound_chord_m": chord,
            },
            cost=float(self.reposition_fixed_cost_m + self.reposition_path_weight * chord),
        )

    def validate_lazy_candidate(self, path: Sequence[PlannerEdge]) -> tuple[list[PlannerEdge], str]:
        """Run A* then replay/cache for each reposition in one goal candidate."""
        validated: list[PlannerEdge] = []
        for edge in path:
            if edge.edge_type != "REPOSITION" or not edge.checks.get("lazy_candidate_pending"):
                validated.append(edge)
                continue
            if edge.source is None or edge.target_face is None:
                return [], "MALFORMED_LAZY_REPOSITION_EDGE"
            exact = self.reposition_successor(edge.source, edge.target_face)
            if exact is None:
                self._forbidden_lazy_repositions.add(self._lazy_reposition_identity(edge.source, edge.target_face))
                return [], f"LAZY_REPOSITION_VALIDATION_FAIL:{edge.edge_id}"
            validated.append(exact)
        return validated, ""

    def terminal_goal_connectors(
        self,
        state: PlannerState,
        goal: BoxPose,
        primitive_families: Sequence[PushPrimitive],
        *,
        max_length_m: float = 0.70,
        position_tolerance_m: float = 1e-6,
        yaw_tolerance_rad: float = 1e-6,
    ) -> list[PlannerEdge]:
        """Generate an arbitrary short length only when it directly reaches the goal."""
        result: list[PlannerEdge] = []
        seen: set[tuple[str, int]] = set()
        for prototype in primitive_families:
            if prototype.source_face != state.face:
                continue
            family_key = (prototype.family_id, int(round(prototype.curvature_m_inv * 1e9)))
            if family_key in seen:
                continue
            seen.add(family_key)
            kappa = float(prototype.curvature_m_inv)
            if abs(kappa) <= 1e-10:
                if abs(wrap_angle(goal.yaw - state.q_box.yaw)) > yaw_tolerance_rad:
                    continue
                local_delta = _rotation(-state.q_box.yaw) @ (goal.xy - state.q_box.xy)
                direction = np.asarray(prototype.direction_local, dtype=np.float64)
                direction /= np.linalg.norm(direction)
                length = float(local_delta @ direction)
                cross = float(np.linalg.norm(local_delta - length * direction))
                if cross > position_tolerance_m:
                    continue
            else:
                yaw_delta = wrap_angle(goal.yaw - state.q_box.yaw)
                length = yaw_delta / kappa
                if length <= 0.0:
                    continue
            if not (1e-9 < length <= max_length_m + 1e-12):
                continue
            connector = PushPrimitive(
                primitive_id=f"TERMINAL_GOAL_CONNECTOR_{prototype.family_id}_{length:.6f}",
                source_face=prototype.source_face,
                length_m=length,
                curvature_m_inv=kappa,
                samples=max(3, int(math.ceil(length / 0.025)) + 1),
                direction_local=prototype.direction_local,
                family=prototype.family_id,
            )
            poses = sample_box_poses(state.q_box, connector)
            if np.linalg.norm(poses[-1].xy - goal.xy) > position_tolerance_m:
                continue
            if abs(wrap_angle(poses[-1].yaw - goal.yaw)) > yaw_tolerance_rad:
                continue
            edge = self.push_successor(state, connector, terminal_goal_connector=True)
            if edge is not None:
                result.append(edge)
        return result

    def state_key(self, state: PlannerState) -> tuple[int, int, int, int, str | None]:
        """Closed key ``(kappa(q), lambda, previous_primitive)``."""
        return (
            int(round(state.q_box.x / self.state_xy_resolution_m)),
            int(round(state.q_box.y / self.state_xy_resolution_m)),
            int(round(wrap_angle(state.q_box.yaw) / self.state_yaw_resolution_rad)),
            int(state.interaction_lambda),
            state.previous_primitive,
        )

    def search(
        self,
        *,
        start_box: BoxPose,
        start_face: str | None,
        initial_template_mode: str,
        primitives: Sequence[PushPrimitive],
        goal_test: Callable[[PlannerState], bool] | None = None,
        goal_pose: BoxPose | None = None,
        goal_position_tolerance_m: float = 0.04,
        goal_yaw_tolerance_rad: float = 0.03,
        terminal_connector_max_m: float = 0.70,
        max_expansions: int = 512,
        heuristic: Callable[[PlannerState], float] | None = None,
        weighted_astar_weight: float = 1.25,
        lazy_reposition_validation: bool = True,
        max_lazy_restarts: int = 3,
        _lazy_restart_depth: int = 0,
    ) -> tuple[list[PlannerEdge], dict[str, object]]:
        if initial_template_mode not in ("FIXED_INITIAL_TEMPLATE", "FREE_INITIAL_TEMPLATE"):
            raise ValueError(initial_template_mode)
        if weighted_astar_weight < 1.0:
            raise ValueError("weighted_astar_weight must be >= 1")
        if goal_test is None:
            if goal_pose is None:
                raise ValueError("goal_test or goal_pose is required")

            def goal_test(state: PlannerState) -> bool:
                return bool(
                    np.linalg.norm(state.q_box.xy - goal_pose.xy) <= goal_position_tolerance_m
                    and abs(wrap_angle(state.q_box.yaw - goal_pose.yaw)) <= goal_yaw_tolerance_rad
                )
        faces = (start_face,) if initial_template_mode == "FIXED_INITIAL_TEMPLATE" else FACE_NAMES
        if any(face not in FACE_NAMES for face in faces):
            return [], {"passed": False, "reason": "INVALID_INITIAL_FACE", "expanded_nodes": 0}
        frontier: list[tuple[float, int, float, PlannerState, list[PlannerEdge]]] = []
        serial = 0
        generated = 0
        for face in faces:
            valid, _, base_path = self._initial_stance_ok(start_box, face)
            if not valid:
                continue
            state = PlannerState(start_box, FACE_LAMBDA[face], None)
            start_length = 0.0 if base_path is None else base_path.length_m
            edge = PlannerEdge(
                edge_id=f"START_{face}", edge_type="START", source=None, target=state,
                target_face=face, base_path_xy=[] if base_path is None else list(base_path.path_xy),
                checks={
                    "virtual_start": initial_template_mode == "FREE_INITIAL_TEMPLATE",
                    "stance_validity_pass": True,
                    "base_accessibility_pass": True,
                    "lambda": FACE_LAMBDA[face],
                },
                cost=self.reposition_path_weight * start_length,
            )
            h_value = 0.0 if heuristic is None else max(0.0, float(heuristic(state)))
            heapq.heappush(frontier, (edge.cost + weighted_astar_weight * h_value, serial, edge.cost, state, [edge]))
            serial += 1
            generated += 1
        expanded = 0
        best_g: dict[tuple[int, int, int, int, str | None], float] = {}
        attempted_push = attempted_reposition = 0
        generated_push = generated_reposition = generated_terminal = 0
        while frontier and expanded < max_expansions:
            _, _, g_cost, state, path = heapq.heappop(frontier)
            key = self.state_key(state)
            if g_cost >= best_g.get(key, math.inf) - 1e-12:
                continue
            best_g[key] = g_cost
            expanded += 1
            if goal_test(state):
                final_path = path
                lazy_reason = ""
                if lazy_reposition_validation:
                    final_path, lazy_reason = self.validate_lazy_candidate(path)
                if lazy_reason:
                    if _lazy_restart_depth >= max_lazy_restarts:
                        return [], {
                            "passed": False,
                            "reason": "LAZY_REPOSITION_RESTART_LIMIT",
                            "expanded_nodes": expanded,
                            "generated_successors": generated,
                            "lazy_goal_candidates_rejected": 1,
                            "lazy_rejection_reasons": [lazy_reason],
                            "initial_template_mode": initial_template_mode,
                            "weighted_astar_weight": weighted_astar_weight,
                        }
                    replacement, replacement_audit = self.search(
                        start_box=start_box,
                        start_face=start_face,
                        initial_template_mode=initial_template_mode,
                        primitives=primitives,
                        goal_test=goal_test,
                        goal_pose=goal_pose,
                        goal_position_tolerance_m=goal_position_tolerance_m,
                        goal_yaw_tolerance_rad=goal_yaw_tolerance_rad,
                        terminal_connector_max_m=terminal_connector_max_m,
                        max_expansions=max_expansions,
                        heuristic=heuristic,
                        weighted_astar_weight=weighted_astar_weight,
                        lazy_reposition_validation=lazy_reposition_validation,
                        max_lazy_restarts=max_lazy_restarts,
                        _lazy_restart_depth=_lazy_restart_depth + 1,
                    )
                    replacement_audit["expanded_nodes"] = int(replacement_audit.get("expanded_nodes", 0)) + expanded
                    replacement_audit["generated_successors"] = int(replacement_audit.get("generated_successors", 0)) + generated
                    replacement_audit["lazy_goal_candidates_rejected"] = int(
                        replacement_audit.get("lazy_goal_candidates_rejected", 0)
                    ) + 1
                    replacement_audit.setdefault("lazy_rejection_reasons", []).insert(0, lazy_reason)
                    return replacement, replacement_audit
                exact_cost = float(sum(edge.cost for edge in final_path))
                return final_path, {
                    "passed": True,
                    "expanded_nodes": expanded,
                    "generated_successors": generated,
                    "generated_push": generated_push,
                    "generated_reposition": generated_reposition,
                    "generated_terminal_connectors": generated_terminal,
                    "attempted_push": attempted_push,
                    "attempted_reposition": attempted_reposition,
                    "closed_states": len(best_g),
                    "initial_template_mode": initial_template_mode,
                    "weighted_astar_weight": weighted_astar_weight,
                    "reposition_cache_hits": self.reposition_cache_hits,
                    "reposition_cache_misses": self.reposition_cache_misses,
                    "path_cost_m_equivalent": exact_cost,
                    "lazy_reposition_validation": lazy_reposition_validation,
                }
            candidates: list[PlannerEdge] = []
            for primitive in primitives:
                attempted_push += 1
                edge = self.push_successor(state, primitive)
                if edge is not None:
                    candidates.append(edge)
                    generated_push += 1
            if goal_pose is not None:
                connectors = self.terminal_goal_connectors(
                    state,
                    goal_pose,
                    primitives,
                    max_length_m=terminal_connector_max_m,
                    position_tolerance_m=goal_position_tolerance_m,
                    yaw_tolerance_rad=goal_yaw_tolerance_rad,
                )
                candidates.extend(connectors)
                generated_terminal += len(connectors)
            for face in FACE_NAMES:
                attempted_reposition += 1
                edge = self.lazy_reposition_successor(state, face) if lazy_reposition_validation else self.reposition_successor(state, face)
                if edge is not None:
                    candidates.append(edge)
                    generated_reposition += 1
            for edge in candidates:
                if edge.target is None:
                    continue
                next_g = g_cost + edge.cost
                if next_g >= best_g.get(self.state_key(edge.target), math.inf) - 1e-12:
                    continue
                h_value = 0.0 if heuristic is None else max(0.0, float(heuristic(edge.target)))
                heapq.heappush(
                    frontier,
                    (next_g + weighted_astar_weight * h_value, serial, next_g, edge.target, path + [edge]),
                )
                serial += 1
                generated += 1
        return [], {
            "passed": False,
            "expanded_nodes": expanded,
            "generated_successors": generated,
            "generated_push": generated_push,
            "generated_reposition": generated_reposition,
            "generated_terminal_connectors": generated_terminal,
            "attempted_push": attempted_push,
            "attempted_reposition": attempted_reposition,
            "closed_states": len(best_g),
            "reason": "SEARCH_NO_ACCESS_AWARE_PLAN" if not frontier else "SEARCH_EXPANSION_LIMIT",
            "initial_template_mode": initial_template_mode,
            "weighted_astar_weight": weighted_astar_weight,
            "reposition_cache_hits": self.reposition_cache_hits,
            "reposition_cache_misses": self.reposition_cache_misses,
        }


def edge_sequence_summary(edges: Iterable[PlannerEdge]) -> list[dict[str, object]]:
    return [
        {
            "edge_id": edge.edge_id, "type": edge.edge_type,
            "source_face": edge.source_face, "target_face": edge.target_face,
            "source_lambda": None if edge.source is None else edge.source.interaction_lambda,
            "target_lambda": None if edge.target is None else edge.target.interaction_lambda,
            "source_q": None if edge.source is None else edge.source.q_box,
            "target_q": None if edge.target is None else edge.target.q_box,
            "previous_primitive_before": None if edge.source is None else edge.source.previous_primitive,
            "previous_primitive_after": None if edge.target is None else edge.target.previous_primitive,
            "box_pose_count": len(edge.box_poses), "base_waypoint_count": len(edge.base_path_xy),
            "cost": edge.cost,
            "checks": edge.checks,
        }
        for edge in edges
    ]
