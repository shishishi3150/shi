"""Task success is evaluated independently from reward."""
from dataclasses import dataclass, field
import math
import numpy as np


def encirclement_geometry(target, positions):
    """Measure actual enclosure, without assigned vertices or desired points."""
    offsets = np.asarray(positions, dtype=float).reshape(-1, 2) - np.asarray(target, dtype=float)
    radii = np.linalg.norm(offsets, axis=1)
    if len(offsets) < 3:
        return dict(radii_m=radii.tolist(), maximum_angle_gap_rad=2 * math.pi, hull_margin_m=0.)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    order = np.argsort(angles)
    angles = angles[order]
    vertices = offsets[order]
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    gap = float(np.max(np.diff(np.r_[angles, angles[0] + 2 * math.pi])))
    # Sorted bearings put an enclosing polygon counterclockwise. A target
    # outside it has a negative signed distance to at least one edge.
    margins = (edges[:, 1] * vertices[:, 0] - edges[:, 0] * vertices[:, 1]) / np.maximum(lengths, 1e-8)
    margin = float(np.min(margins)) if np.all(lengths > 1e-8) else 0.
    return dict(radii_m=radii.tolist(), maximum_angle_gap_rad=gap, hull_margin_m=margin)


@dataclass
class TaskState:
    hold_steps: int = 0
    engaged_once: bool = False
    gates: dict = field(default_factory=dict)
    support_gates: dict = field(default_factory=dict)

    def advance(self, *, config, dt, red_x, engaged_usvs, shadow_gap, progress_deficit,
                observers, angle_gap, informed_usvs, route_available, red_intent, safe,
                capture_distances, capture_hull_margin, counterfactual_valid=False):
        left,right=config['engagement_x_range']
        engagement=left<=red_x<=right and engaged_usvs>=1
        self.engaged_once |= engagement
        self.support_gates={
            'engagement':engagement and self.engaged_once,
            'causal_blocking':shadow_gap>=config['minimum_shadow_gap_m'] and progress_deficit>=config['minimum_progress_deficit_m'],
            'current_state_counterfactual':bool(counterfactual_valid),
            'information_support':informed_usvs>=config['minimum_informed_usvs'],
            'red_breakthrough_intent':bool(red_intent)}
        distances = np.asarray(capture_distances, dtype=float)
        self.gates={
            'auv_participation':len(distances)==3,
            'capture_distance':bool(len(distances)==3 and np.all(distances>=config['capture_min_radius_m'])
                                    and np.all(distances<=config['capture_max_radius_m'])),
            'encirclement':bool(angle_gap<=math.radians(config['maximum_observer_angle_gap_degrees'])
                                and capture_hull_margin>=config['capture_min_hull_margin_m']),
            'observation_support':observers>=config['minimum_auv_observers'],
            'route_available':bool(route_available), 'safety':bool(safe)}
        self.hold_steps=self.hold_steps+1 if all(self.gates.values()) else 0
        return self.hold_steps>=math.ceil(config['success_hold_seconds']/dt)
