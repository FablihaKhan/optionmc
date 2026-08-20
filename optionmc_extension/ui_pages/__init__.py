"""One module per dashboard page.

Named `ui_pages` rather than `pages` on purpose: Streamlit treats a top-level
`pages/` directory as an implicit navigation source, which would fight the
explicit `st.navigation` wiring in `app.py` and put every page in the sidebar
twice.
"""
