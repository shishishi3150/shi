"""Read existing run timings without loading a model or changing training."""
import argparse
import json
import math
from pathlib import Path


def summarize(manifest):
    runtime = manifest.get('runtime') or {}
    wall = float(runtime.get('wall_time_s', manifest.get('wall_time_s', 0.)))
    steps = int(manifest.get('online_steps', 0))
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError('manifest has no positive runtime.wall_time_s')
    phases = runtime.get('phase_seconds') or {}
    if not phases:
        raise ValueError('manifest has no runtime.phase_seconds; use a recent runtime record')
    rows = []
    for name, raw in phases.items():
        seconds = float(raw)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'invalid timing: {name}')
        rows.append(dict(phase=name, seconds=seconds, percent_of_wall=100*seconds/wall,
            amortized_ms_per_online_step=1000*seconds/steps if steps else None))
    other = max(0., wall-sum(row['seconds'] for row in rows))
    rows.append(dict(phase='other', seconds=other, percent_of_wall=100*other/wall,
        amortized_ms_per_online_step=1000*other/steps if steps else None))
    rows.sort(key=lambda row: row['seconds'], reverse=True)
    config = manifest.get('config') or {}
    training = config.get('training') or {}
    diagnostics = manifest.get('diagnostic_totals') or {}
    updates = int(runtime.get('update_count', 0))
    update_seconds = float(phases.get('update', 0.))
    nested = {}
    for name in ('i1_target_vo_time_s', 'i1_actor_vo_time_s', 'i3_vo_screen_time_s'):
        if name in diagnostics:
            seconds = float(diagnostics[name])
            nested[name] = dict(seconds=seconds,
                percent_of_update=100*seconds/update_seconds if update_seconds else None)
    return dict(online_steps=steps, completed_episodes=manifest.get('completed_episodes'),
        wall_time_hours=wall/3600., online_steps_per_second=steps/wall,
        actual_device=runtime.get('actual_device'), hardware=manifest.get('hardware'),
        phase_breakdown=rows, update_count=updates,
        update_ms=1000*update_seconds/updates if updates else None,
        nested_update_timings=nested,
        timing_note='Nested I1/I3 timings are already inside update; do not add them again. '
                    'CUDA phase timing follows the existing trainer and is not a kernel profiler.',
        configuration=dict(experiment=(config.get('revision') or {}).get('experiment'),
            batch_size=training.get('batch_size'), hidden_sizes=training.get('hidden_sizes'),
            evaluation=config.get('evaluation'), i3=config.get('i3'),
            diagnostics=config.get('diagnostics')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path,
        help='training run directory, or its run_manifest.json')
    args = parser.parse_args(argv)
    path = args.run/'run_manifest.json' if args.run.is_dir() else args.run
    try:
        manifest = json.loads(path.read_text(encoding='utf-8-sig'))
        result = summarize(manifest)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(manifest=str(path.resolve()), **result), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
