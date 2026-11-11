"""List or restore byte-identical cold validation evidence; never overwrite.

Examples:
  python -B scripts/restore_validation_archive.py --list
  python -B scripts/restore_validation_archive.py --path validation/old_run
  python -B scripts/restore_validation_archive.py --all
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'validation/archive_index.json'


def validation_path(value):
    relative = PurePosixPath(value)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in value
            or ':' in value or not relative.parts or relative.parts[0] != 'validation'):
        raise ValueError('archive paths must stay inside validation')
    target = (ROOT / Path(*relative.parts)).resolve()
    if not target.is_relative_to(ROOT / 'validation'):
        raise ValueError('archive path escapes validation')
    return target


def restore(row):
    target = validation_path(row['path'])
    archive = validation_path(row['archive'])
    if row['member'] != row['path']:
        raise ValueError('archive member differs from the indexed original path')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise FileExistsError(f'refusing to overwrite {target}')
        return False
    with zipfile.ZipFile(archive) as source:
        data = source.read(row['member'])
    if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('archived bytes failed integrity verification')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list', action='store_true')
    mode.add_argument('--all', action='store_true')
    mode.add_argument('--path', action='append', help='original file or directory prefix; repeatable')
    args = parser.parse_args(argv)
    rows = json.loads(INDEX.read_text(encoding='utf-8'))['entries']
    if args.list:
        for row in rows:
            print(f"{row['bytes']:>10}  {row['path']}")
        return len(rows)
    prefixes = [Path(value).as_posix().rstrip('/') for value in (args.path or [])]
    for value in prefixes:
        validation_path(value)
    selected = rows if args.all else [row for row in rows if any(
        row['path'] == prefix or row['path'].startswith(prefix + '/') for prefix in prefixes)]
    if not selected:
        parser.error('no archived evidence matched the requested path')
    count = sum(restore(row) for row in selected)
    print(f'Restored {count} files; {len(selected)-count} already matched. No files overwritten.')
    return count


if __name__ == '__main__':
    main()
