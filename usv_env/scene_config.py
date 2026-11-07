"""Geometry checks in physical metres for the single current scene."""
import math
import numpy as np
from .geometry import capsule_segment,capsule_circle_clearance,capsule_inside_rectangle,capsules_overlap


def validate_scene(config, expected=None):
    world=np.asarray(config['env']['world_size'],float)
    reefs=config['reefs']; depth=config['depth']; ship=config['ship']; scene=config['scenario']
    if world.shape!=(2,) or not np.isfinite(world).all() or min(world)<=0: raise ValueError('invalid world size')
    for reef in reefs:
        center=np.asarray(reef['center'],float); radius=float(reef['radius'])+depth['reef_shallow_buffer_m']
        if center.shape!=(2,) or not np.isfinite(center).all() or radius<=0: raise ValueError('invalid reef')
        if np.any(center-radius<0) or np.any(center+radius>world): raise ValueError('reef exclusion contour outside world')
    for i,left in enumerate(reefs):
        for right in reefs[i+1:]:
            if np.linalg.norm(np.asarray(left['center'])-right['center'])<=left['radius']+right['radius']+2*depth['reef_shallow_buffer_m']: raise ValueError('reef exclusion contours overlap')
    records=[]
    for role,positions,headings in [('surface',scene['blue_starts'],scene['blue_headings']),('submarine',scene['submarine_starts'],scene['submarine_headings']),('surface',[scene['red_start']],[scene['red_heading']])]:
        length,width=(ship['submarine_collision_length_m'],ship['submarine_collision_width_m']) if role=='submarine' else (ship['surface_length_m'],ship['surface_collision_width_m'])
        for pos,heading in zip(positions,headings):
            capsule=capsule_segment(pos,heading,length,width)
            if not capsule_inside_rectangle(capsule,world): raise ValueError('initial hull outside boundary')
            if any(capsule_circle_clearance(capsule,np.asarray(r['center']),r['radius']+depth['reef_shallow_buffer_m'])<=0 for r in reefs): raise ValueError('initial hull overlaps reef/shallow contour')
            records.append((role,capsule))
    separated=depth['auv_depth_m']-depth['surface_keel_depth_m']>depth['auv_vertical_radius_m']+depth['vertical_safety_margin_m']
    for i,(role,left) in enumerate(records):
        for other,right in records[i+1:]:
            if (role==other or not separated) and capsules_overlap(left,right): raise ValueError('initial hulls overlap at collision depth')
    zone=config['protected_zone']; center=np.asarray(zone['center'],float); radius=zone['radius']
    if np.any(center-radius<0) or np.any(center+radius>world): raise ValueError('protected zone outside world')
    if any(np.linalg.norm(center-np.asarray(r['center']))<=radius+r['radius']+depth['reef_shallow_buffer_m'] for r in reefs): raise ValueError('protected zone overlaps reef')
    return {'initial_hulls':len(records),'reefs':len(reefs),'cross_depth_separated':separated}
