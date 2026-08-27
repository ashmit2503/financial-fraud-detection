from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_overview_starts_without_exceptions() -> None:
    entrypoint = Path(__file__).resolve().parent.parent / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=15).run()

    assert not app.exception
    assert any("Fraud model monitor" in title.value for title in app.title)


def test_all_streamlit_pages_render_without_exceptions() -> None:
    entrypoint = Path(__file__).resolve().parent.parent / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=15).run()

    for page in (
        "app_pages/performance.py",
        "app_pages/drift.py",
        "app_pages/diagnosis.py",
    ):
        app.switch_page(page).run()
        assert not app.exception, page
