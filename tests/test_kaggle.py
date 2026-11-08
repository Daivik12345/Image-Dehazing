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
        assert not any(name.endswith("/.DS_Store") for name in names)
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


def test_notebook_reruns_preserve_existing_images_and_logs(tmp_path):
    notebook = json.loads((ROOT / "notebooks/btp-baseline-plus.ipynb").read_text())
    functions = [
        node
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef) and node.name == "available_case_name"
    ]
    assert len(functions) == 1
    scope = {"OUTPUT": tmp_path}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "notebook", "exec"), scope
    )
    choose = scope["available_case_name"]
    assert choose("01-hazegen") == "01-hazegen"
    (tmp_path / "01-hazegen").mkdir()
    (tmp_path / "01-hazegen" / "keep.txt").write_text("previous result")
    (tmp_path / "01-hazegen-rerun-001.log").write_text("previous failure")
    assert choose("01-hazegen") == "01-hazegen-rerun-002"
    assert (tmp_path / "01-hazegen" / "keep.txt").read_text() == "previous result"


def test_evaluation_notebook_is_separate_and_compiles():
    notebook = json.loads((ROOT / "notebooks/btp-eval-optimised.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "btp-eval-source.bundle" in code
    assert "nh-haze-512.bundle" in code
    assert "btp.benchmark" in code
    assert 'run_evaluation("pilot")' in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
