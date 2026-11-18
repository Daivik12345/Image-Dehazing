"""Archive inventory must reject unsafe or ambiguous evidence."""

import zipfile

import pytest

from btp.evidence import audit_archive


def test_archive_inventory(tmp_path):
    p = tmp_path / "run.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("stage/jobs.jsonl", '{"job_id":"a","status":"completed"}\n')
    result = audit_archive(p)
    assert result["stages"]["stage"]["statuses"] == {"completed": 1}
    assert len(result["archive_sha256"]) == 64


@pytest.mark.parametrize("name", ["../escape", "/absolute", "stage/../../escape"])
def test_archive_rejects_paths(tmp_path, name):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name, "x")
    with pytest.raises(ValueError, match="Unsafe"):
        audit_archive(p)


def test_duplicate_jobs_rejected(tmp_path):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("stage/jobs.jsonl", '{"job_id":"a","status":"completed"}\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        audit_archive(p)
