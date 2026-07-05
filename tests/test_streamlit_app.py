import py_compile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app" / "app.py"


def test_streamlit_app_compiles():
    py_compile.compile(str(APP_PATH), doraise=True)


def test_streamlit_app_uses_current_width_api():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "use_container_width" not in source
    assert 'width="stretch"' in source


def test_streamlit_navigation_is_single_click_friendly():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "st.query_params.get" in source
    assert "st.query_params[" not in source
    assert "PAGE_LABELS" in source
    assert "Executive Fraud Overview" in source
    assert "Explainable AI" in source


def test_streamlit_display_table_omits_none_height():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "if height is not None:" in source
    assert 'options["height"] = height' in source