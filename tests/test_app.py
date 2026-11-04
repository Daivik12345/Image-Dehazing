"""Exercise the interactive viewer through Streamlit's real test harness."""

from streamlit.testing.v1 import AppTest

from btp.config import ROOT


def test_live_viewer_can_run_cpu_pipeline():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.success
    assert "Saved experiment" in app.success[0].value


def test_pretrained_run_is_disabled_without_weights(tmp_path):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.text_input[0].set_value(str(tmp_path)).run()
    app.selectbox[0].select("Pretrained paper models").run()
    assert not app.exception
    assert app.button[0].disabled


def test_observer_rejects_outside_paths():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.query_params["run"] = "../AGENTS.md"
    app.run()
    assert not app.exception
    assert "Unknown experiment" in app.error[0].value


def test_observer_shows_completed_experiment():
    from btp.config import RunConfig
    from btp.experiment import create_run, run_preview
    from btp.preview import demo_image

    config = RunConfig(size=128, steps=2)
    image = demo_image(128)
    run = create_run(image, config)
    run_preview(image, config, run)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.query_params["run"] = run.name
    app.run()
    assert not app.exception
    assert "Experiment completed" in app.success[0].value
