# Streamlit Pages

This project intentionally uses a single-page Streamlit application with custom internal navigation in `streamlit_app/app.py`.

The folder is retained for governance assets and future migration planning. No `.py` files are placed here because native Streamlit multipage files would create a second sidebar navigation and reduce dashboard polish.

Current page metadata is documented in `page_manifest.json`.

Direct URLs supported by the app:

- `/?page=executive-fraud-overview`
- `/?page=transaction-intelligence`
- `/?page=fraud-pattern-analysis`
- `/?page=model-performance`
- `/?page=fraud-risk-scoring`
- `/?page=explainable-ai`