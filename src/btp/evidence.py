"""Read-only, bounded auditing of exported experiment archives."""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from btp.evaluation import sha256


def audit_archive(path: Path, max_bytes: int = 2 * 1024**3) -> dict[str, Any]:
    """Inventory a ZIP without extracting it or treating summaries as approval.

    This verifies archive CRCs and job identities, not scientific eligibility.
    Call the optimization validators on predictions before selecting a policy.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    stages = {}
    with zipfile.ZipFile(path) as archive:
        seen = set()
        total = 0
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if item.filename in seen:
                raise ValueError(f"Duplicate archive member: {item.filename}")
            seen.add(item.filename)
            total += item.file_size
            if total > max_bytes:
                raise ValueError("Archive exceeds uncompressed size budget")
            if name.name != "jobs.jsonl":
                continue
            if item.file_size > 16 * 1024**2:
                raise ValueError("Job inventory exceeds size budget")
            rows = [json.loads(line) for line in archive.read(item).splitlines()]
            ids = [row["job_id"] for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate job identities")
            stages[str(name.parent)] = {
                "jobs": len(rows),
                "statuses": dict(Counter(row["status"] for row in rows)),
            }
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Archive CRC failure: {bad}")
    return {
        "archive_sha256": sha256(path),
        "files": len(seen),
        "uncompressed_bytes": total,
        "stages": stages,
        "scope": "Inventory and CRC validation only; not scientific acceptance",
    }


def main() -> None:
    """Print a portable evidence inventory without modifying the archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.archive), indent=2))


if __name__ == "__main__":
    main()
