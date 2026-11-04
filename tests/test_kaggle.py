"""Check portable source packaging and unexecuted notebook syntax."""

import ast
import hashlib
import json
import zipfile

from btp.config import ROOT
from btp.kaggle import build_bundle


def test_source_bundle_contains_upstream_but_no_weights_or_credentials(tmp_path):
    archive = tmp_path / "source.zip"
    build_bundle(archive)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert "BTP/upstream/Learning-Hazing-to-Dehazing/train_stage1.py" in names
        assert "BTP/upstream/Learning-Hazing-to-Dehazing/inference_accsamp.py" in names
        assert "BTP/notebooks/kaggle_reference.ipynb" in names
        assert not any(
            "/weights/" in n or "/.git/" in n or "/.venv" in n or "/.env" in n
            for n in names
        )
        snapshot = json.loads(bundle.read("BTP/source-manifest.json"))
        for name, digest in snapshot["files"].items():
            assert hashlib.sha256(bundle.read("BTP/" + name)).hexdigest() == digest


def test_notebook_cells_are_unexecuted_and_parse():
    notebook = json.loads((ROOT / "notebooks/kaggle_reference.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
