# Dashboard Design Notes

The dashboard uses a single Streamlit entry point (`streamlit_app/app.py`) with internal navigation. This avoids duplicate Streamlit multipage navigation and keeps the portfolio review flow controlled.

Design principles:

- Light analytical canvas using `#D6E4F0`
- High-contrast dark text on light cards
- Navy sidebar for stable navigation
- Risk colors reserved for fraud severity
- Tables show selected analyst-facing columns instead of raw wide exports
- Charts use explicit axis titles, readable heights, and no empty log-scale rendering
- Navigation is single-click radio navigation. Query parameters are read only for direct page links and are not rewritten during click events.
- Tables omit `height` unless a fixed table viewport is intentionally needed, matching Streamlit's Python 3.13.1 environment.
- Current Streamlit width API uses `width="stretch"`, not deprecated `use_container_width`.