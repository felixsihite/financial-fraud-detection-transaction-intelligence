"""Consistent visual theme for charts and Streamlit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import APPROVED_PALETTE


def set_plot_theme() -> None:
    """Apply a professional financial-risk visual style."""

    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": APPROVED_PALETTE["light_background"],
            "axes.facecolor": APPROVED_PALETTE["card_background"],
            "axes.edgecolor": APPROVED_PALETTE["secondary_text"],
            "axes.labelcolor": APPROVED_PALETTE["primary_text"],
            "xtick.color": APPROVED_PALETTE["secondary_text"],
            "ytick.color": APPROVED_PALETTE["secondary_text"],
            "text.color": APPROVED_PALETTE["primary_text"],
            "grid.color": "#CBD5E1",
            "font.size": 10,
        },
    )
    plt.rcParams["axes.titleweight"] = "bold"


STREAMLIT_CSS = f"""
<style>
:root {{
  --bg: {APPROVED_PALETTE["light_background"]};
  --surface: {APPROVED_PALETTE["card_background"]};
  --surface-2: {APPROVED_PALETTE["secondary_surface"]};
  --text: {APPROVED_PALETTE["primary_text"]};
  --muted: {APPROVED_PALETTE["secondary_text"]};
  --navy: {APPROVED_PALETTE["primary_navy"]};
  --blue: {APPROVED_PALETTE["accent_blue"]};
  --teal: {APPROVED_PALETTE["accent_teal"]};
  --green: {APPROVED_PALETTE["positive_green"]};
  --amber: {APPROVED_PALETTE["warning_amber"]};
  --red: {APPROVED_PALETTE["risk_red"]};
  --critical: {APPROVED_PALETTE["critical_red"]};
  --border: #BFD7EA;
  --grid: #D8E5EF;
  --shadow: 0 10px 28px rgba(11, 31, 51, 0.08);
}}

.stApp {{
  background: var(--bg);
  color: var(--text);
}}

[data-testid="stSidebar"] {{
  background: var(--navy);
  border-right: 1px solid rgba(255, 255, 255, 0.10);
  min-width: 18rem;
  max-width: 18rem;
}}

[data-testid="stSidebar"] * {{
  color: #F8FAFC;
}}

[data-testid="stSidebar"] .stRadio label {{
  padding: 0.20rem 0;
}}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label p {{
  font-size: 0.92rem;
  line-height: 1.25;
}}

.block-container {{
  padding: 1.05rem 1.35rem 2.5rem !important;
  max-width: 720px !important;
}}

h1, h2, h3 {{
  color: var(--text);
  letter-spacing: 0;
}}

h1,
[data-testid="stHeading"] h1,
[data-testid="stMarkdownContainer"] h1 {{
  font-size: 1.68rem !important;
  line-height: 1.12 !important;
  font-weight: 800 !important;
  margin-bottom: 0.65rem !important;
  max-width: 680px !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
}}

h2 {{
  font-size: 1.35rem;
  margin-top: 1.2rem;
}}

h3 {{
  font-size: 1.08rem;
}}

p, li, label, span, div {{
  color: inherit;
}}

.page-kicker {{
  color: var(--teal);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}}

.section-title {{
  color: var(--text);
  font-size: 1.05rem;
  font-weight: 800;
  margin: 1.2rem 0 .55rem;
}}

.metric-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  min-height: 112px;
  box-shadow: var(--shadow);
}}

.metric-label {{
  color: var(--muted);
  font-size: 0.74rem;
  font-weight: 700;
  text-transform: uppercase;
}}

.metric-value {{
  color: var(--text);
  font-size: 1.34rem;
  line-height: 1.15;
  font-weight: 800;
  margin-top: 8px;
  overflow-wrap: anywhere;
}}

.metric-note {{
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.35;
  margin-top: 9px;
}}

.insight-box {{
  background: var(--surface);
  border-left: 5px solid var(--teal);
  border-radius: 8px;
  color: var(--text);
  padding: 14px 16px;
  margin: 12px 0 18px 0;
  box-shadow: var(--shadow);
}}

.status-band {{
  background: linear-gradient(90deg, #F7FAFC 0%, #EAF3FA 100%);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 13px 16px;
  color: var(--text);
  box-shadow: var(--shadow);
}}

.risk-pill {{
  display: inline-block;
  border-radius: 999px;
  color: #FFFFFF;
  font-size: 0.72rem;
  font-weight: 800;
  padding: 4px 9px;
  text-transform: uppercase;
}}

.risk-critical {{
  background: var(--critical);
}}

.risk-high {{
  background: var(--red);
}}

.risk-medium {{
  background: var(--amber);
}}

.risk-low {{
  background: var(--green);
}}

[data-testid="stDataFrame"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: var(--shadow);
}}

[data-testid="stImage"] img {{
  background: var(--surface);
  border-radius: 8px;
}}

.stButton button,
.stDownloadButton button {{
  background: var(--blue);
  color: #FFFFFF;
  border: 1px solid var(--blue);
  border-radius: 8px;
  font-weight: 800;
  min-height: 2.65rem;
}}

.stButton button:hover,
.stDownloadButton button:hover {{
  background: #1D4ED8;
  color: #FFFFFF;
  border-color: #1D4ED8;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px 8px 0 0;
  color: var(--text);
}}

.stTabs [aria-selected="true"] {{
  background: var(--navy);
  color: #FFFFFF;
}}

hr {{
  border-color: var(--border);
  margin: .8rem 0 1rem;
}}

@media (max-width: 900px) {{
  .block-container {{
    padding: 1rem 1rem 2rem;
  }}
  h1 {{
    font-size: 1.75rem;
  }}
}}
</style>
"""