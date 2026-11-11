"""Inspect the assembled current configuration without fabricating policy outcomes."""
import argparse
import json

from usv_env.config import PRESETS, assemble
from matd3.replay_buffer import ReplayBuffer


def configuration_report(cfg):
    env, training = cfg["env"], cfg["training"]
    count = int(env["num_surface_blue"]) + int(env["num_submarines"])
    schema = cfg["observation_schema"]
    # A single row obtains the actual storage layout without allocating the configured replay.
    layout = ReplayBuffer(1, count, schema["actor_dim"], schema["critic_dim"], 2,
                          n_step_return=training["n_step_return"], gamma=training["gamma"],
                          surface_count=env["num_surface_blue"])
    return dict(task_contract=cfg["task"]["contract"], config_source=cfg["provenance"]["task_file"],
        task_hash=cfg["hashes"]["task"], reward_hash=cfg["hashes"]["reward"],
        code_version=cfg['revision']['version'],i1_method=cfg['i1']['method'],i1_hash=cfg['hashes']['i1'],i1=cfg['i1'],
        world_size_m=env["world_size"], dt_s=env["dt"], max_steps=env["max_steps"],
        episode_seconds=env["dt"]*env["max_steps"], surface_agents=env["num_surface_blue"],
        submarine_agents=env["num_submarines"], learned_agents=count, joint_action_dim=count*2,
        actor_observation_dim=schema["actor_dim"], critic_state_dim=schema["critic_dim"],
        actor_action_units=["surge_force_N", "yaw_moment_Nm"], reefs=cfg["reefs"],
        protected_zone=cfg["protected_zone"], deployment=cfg["scenario"], depth_assumptions=cfg["depth"],
        reward=cfg["reward"], success_conditions=cfg["task"], sensors=cfg["sensors"],
        communication=cfg["communication"], mechanisms=cfg["mechanisms"],
        training=training, replay=dict(capacity=cfg["replay_buffer"]["capacity"],
            save_in_checkpoint=cfg["replay_buffer"]["save_in_checkpoint"],
            resident_array_bytes_per_transition=layout.array_payload_bytes,
            resident_array_bytes=layout.array_payload_bytes*cfg["replay_buffer"]["capacity"],
            estimate_excludes="pending n-step transitions, NumPy/Python overhead and network tensors"),
        effective_n_step_discount=training["gamma"]**training["n_step_return"],
        policy_outputs=["latest.pt", "best.pt"],
        note="Configuration and derived dimensions only; reward quality and task reachability require actual rollout validation.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", choices=PRESETS, default="Base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args=parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    try:
        cfg=assemble(args.config, args.experiment, args.seed, args.device)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    report=configuration_report(cfg)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return report


if __name__ == "__main__":
    main()
