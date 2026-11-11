"""Verify and copy a current actor-only policy without allocating replay."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse
from matd3.checkpoint import read, InferencePolicy, atomic_save

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if Path(args.output).exists(): p.error("output already exists")
    payload = read(args.checkpoint)
    InferencePolicy(payload, "cpu")
    atomic_save(payload, args.output)
    print(f"Saved actor-only policy: {args.output}")

if __name__ == "__main__": main()
