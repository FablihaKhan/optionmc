"""Reusable pieces every page is built from.

Two rules shape this file. The styling leans on Streamlit's own theme
(`.streamlit/config.toml`) and adds only a short sheet of custom CSS under
class names this project owns -- targeting Streamlit's internal class names
would break on the next release and take the presentation with it. And colour
never carries meaning alone: every coloured element also has a label, because
a projector washes out hues and some viewers cannot separate them at all.

The palette is imported from `src/plots.py` so the dashboard and the report
figures cannot drift apart.
"""
import streamlit as st

from src.plots import BENCHMARK, LSMC, MARKET, REFERENCE

NAVY = "#12233a"
INK_SOFT = "#4a5b70"
SURFACE = "#f4f6f9"
BORDER = "#d8dee7"
GOOD = "#1baf7a"
WARN = "#c77700"
BAD = "#c0392b"

_CSS = f"""
<style>
.omc-hero {{
  background: linear-gradient(135deg, {NAVY} 0%, #1d3a5f 100%);
  color: #ffffff;
  padding: 1.6rem 1.9rem;
  border-radius: 12px;
  margin-bottom: 1.1rem;
}}
.omc-hero h1 {{ margin: 0; font-size: 1.85rem; font-weight: 700; color: #fff; }}
.omc-hero p  {{ margin: .35rem 0 0; font-size: 1.02rem; opacity: .92; }}
.omc-hero .omc-caption {{ margin-top: .55rem; font-size: .85rem; opacity: .75; }}

.omc-badge {{
  display: inline-block; padding: .28rem .7rem; border-radius: 999px;
  font-size: .82rem; font-weight: 600; border: 1px solid {BORDER};
  background: {SURFACE}; color: {NAVY}; margin-right: .45rem;
}}
.omc-badge-live {{ background: #fff4e0; border-color: #f0c98a; color: #8a5300; }}

.omc-card {{
  background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px;
  padding: 1rem 1.15rem; height: 100%;
}}
.omc-card h4 {{ margin: 0 0 .2rem; font-size: .82rem; letter-spacing: .04em;
  text-transform: uppercase; color: {INK_SOFT}; font-weight: 700; }}
.omc-card .omc-value {{ font-size: 1.6rem; font-weight: 700; color: {NAVY};
  line-height: 1.15; }}
.omc-card .omc-sub {{ font-size: .85rem; color: {INK_SOFT}; margin-top: .3rem; }}
.omc-card-accent {{ border-left: 4px solid {LSMC}; }}

.omc-note {{
  border-left: 4px solid {LSMC}; background: {SURFACE};
  padding: .8rem 1rem; border-radius: 6px; margin: .6rem 0;
  color: {NAVY}; font-size: .93rem;
}}
.omc-note-warn {{ border-left-color: {WARN}; background: #fff8ec; }}
.omc-note-bad  {{ border-left-color: {BAD};  background: #fdf1ef; }}
.omc-note-good {{ border-left-color: {GOOD}; background: #eefaf5; }}
.omc-note b {{ color: {NAVY}; }}

.omc-hero-metric {{
  background: #ffffff; border: 1px solid {BORDER}; border-top: 5px solid {LSMC};
  border-radius: 10px; padding: 1.1rem 1.3rem; text-align: center;
}}
.omc-hero-metric .omc-label {{ font-size: .86rem; letter-spacing: .05em;
  text-transform: uppercase; color: {INK_SOFT}; font-weight: 700; }}
.omc-hero-metric .omc-big {{ font-size: 3rem; font-weight: 800; color: {LSMC};
  line-height: 1.05; margin: .2rem 0; }}
.omc-hero-metric .omc-sub {{ font-size: .9rem; color: {INK_SOFT}; }}

.omc-footer {{ color: {INK_SOFT}; font-size: .82rem; border-top: 1px solid {BORDER};
  margin-top: 2.2rem; padding-top: .8rem; }}

.omc-swatch {{ display:inline-block; width:.7rem; height:.7rem; border-radius:2px;
  margin-right:.35rem; vertical-align:middle; }}

.omc-pipeline {{ margin:.4rem 0 1rem; }}
.omc-step {{ display:flex; align-items:flex-start; gap:.8rem;
  background:#ffffff; border:1px solid {BORDER}; border-left:4px solid {LSMC};
  border-radius:8px; padding:.6rem .9rem; }}
.omc-step-n {{ flex:0 0 1.6rem; height:1.6rem; border-radius:50%;
  background:{LSMC}; color:#fff; font-weight:700; font-size:.85rem;
  display:flex; align-items:center; justify-content:center; }}
.omc-step-body b {{ color:{NAVY}; }}
.omc-step-detail {{ font-size:.85rem; color:{INK_SOFT}; margin-top:.1rem; }}
.omc-step-arrow {{ text-align:center; color:{LSMC}; font-size:1.1rem;
  line-height:1; margin:.15rem 0; }}

.omc-check {{ display:flex; gap:.75rem; align-items:flex-start;
  padding:.55rem .2rem; border-bottom:1px solid {BORDER}; }}
.omc-check-mark {{ flex:0 0 3.2rem; font-weight:800; font-size:.82rem;
  letter-spacing:.05em; padding-top:.1rem; }}
.omc-check-body b {{ color:{NAVY}; }}
.omc-check-detail {{ font-size:.85rem; color:{INK_SOFT}; margin-top:.1rem; }}
.omc-check-source {{ font-family:ui-monospace, Menlo, Consolas, monospace;
  font-size:.78rem; color:{INK_SOFT}; opacity:.8; }}
</style>
"""


def inject_css():
    """Load the stylesheet once per rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title, subtitle, caption=None):
    """The banner at the top of the app."""
    caption_html = (f'<div class="omc-caption">{caption}</div>'
                    if caption else "")
    st.markdown(
        f'<div class="omc-hero"><h1>{title}</h1><p>{subtitle}</p>'
        f'{caption_html}</div>', unsafe_allow_html=True)


def page_header(title, subtitle=None):
    """A page title with a plain-language line under it.

    The subtitle is not decoration: every page here has a technical name and a
    question it answers, and the question is what a viewer reads first.
    """
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<div style='color:{INK_SOFT}; margin-top:-.5rem; "
                    f"margin-bottom:.9rem;'>{subtitle}</div>",
                    unsafe_allow_html=True)


def timestamp_badge(snapshot, data_mode, live_label="Live market data"):
    """Which data is on screen and how old it is.

    Always visible, because every number on every page is conditional on this
    snapshot and a viewer should never have to guess its date.
    """
    from .formatters import snapshot_caption

    if snapshot is None:
        st.markdown('<span class="omc-badge omc-badge-live">no market '
                    'snapshot on disk</span>', unsafe_allow_html=True)
        return

    caption = snapshot_caption(snapshot.as_of, snapshot.ticker,
                               snapshot.days_to_expiry, snapshot.expiry)
    live = data_mode == live_label
    classes = "omc-badge omc-badge-live" if live else "omc-badge"
    label = "LIVE" if live else "CACHED SNAPSHOT"
    st.markdown(f'<span class="{classes}">{label}</span>'
                f'<span class="omc-badge">{caption}</span>',
                unsafe_allow_html=True)


def metric_card(label, value, sub=None, accent=False):
    """One white card. Kept out of `st.metric` where a subtitle is needed."""
    sub_html = f'<div class="omc-sub">{sub}</div>' if sub else ""
    accent_class = " omc-card-accent" if accent else ""
    st.markdown(
        f'<div class="omc-card{accent_class}"><h4>{label}</h4>'
        f'<div class="omc-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True)


def metric_row(items, columns=None):
    """A row of cards. Never more than four across, per the design rules.

    `items` is a sequence of (label, value) or (label, value, subtitle).
    """
    items = list(items)
    if not items:
        return
    width = columns or min(len(items), 4)
    for start in range(0, len(items), width):
        chunk = items[start:start + width]
        for column, item in zip(st.columns(len(chunk), gap="small"), chunk):
            label, value, *rest = item
            with column:
                metric_card(label, value, rest[0] if rest else None)


def hero_metric(label, value, sub=None):
    """The single number a page is really about."""
    sub_html = f'<div class="omc-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="omc-hero-metric"><div class="omc-label">{label}</div>'
        f'<div class="omc-big">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True)


def callout(text, kind="info"):
    """A short explanatory note. `kind` is info, good, warn or bad."""
    suffix = {"info": "", "good": " omc-note-good",
              "warn": " omc-note-warn", "bad": " omc-note-bad"}[kind]
    st.markdown(f'<div class="omc-note{suffix}">{text}</div>',
                unsafe_allow_html=True)


def what_does_this_mean(lines, title="What does this mean?"):
    """Plain-language reading of the numbers above, shown on request.

    Generated from calculated values by the caller. Nothing here is written by
    a language model at runtime; the sentences are assembled deterministically
    so the same inputs always give the same words.
    """
    if not lines:
        return
    with st.expander(title, expanded=False):
        for line in lines:
            st.markdown(f"- {line}")


def assumption_box(items, title="Model assumptions"):
    """What had to be true for the numbers on this page to mean anything."""
    with st.expander(title, expanded=False):
        for label, value in items:
            st.markdown(f"**{label}** — {value}")


def data_source_box(rows, title="Where this data comes from"):
    with st.expander(title, expanded=False):
        for label, value in rows:
            st.markdown(f"**{label}** — {value}")


def recommendation_card(category, headline, fields, note=None):
    """One optimizer recommendation.

    Never labelled "best": each category answers a different question, and
    which one matters is the investor's choice, not the model's.
    """
    rows = "".join(
        f'<div class="omc-sub"><b>{name}</b>: {value}</div>'
        for name, value in fields)
    note_html = f'<div class="omc-sub" style="margin-top:.45rem">{note}</div>' \
        if note else ""
    st.markdown(
        f'<div class="omc-card omc-card-accent"><h4>{category}</h4>'
        f'<div class="omc-value">{headline}</div>{rows}{note_html}</div>',
        unsafe_allow_html=True)


def warning_card(title, body, kind="warn"):
    callout(f"<b>{title}</b><br>{body}", kind=kind)


def missing_data_notice(missing, needed=None):
    """Say which phase has not been run, instead of drawing an empty chart."""
    if missing is None or len(missing) == 0:
        return False
    if needed is not None:
        missing = missing[missing["item"].isin(needed)]
        if len(missing) == 0:
            return False
    commands = sorted(set(missing["produced_by"]))
    listed = "<br>".join(f"<code>python {command}</code>"
                         for command in commands)
    warning_card("This page needs results that are not on disk yet",
                 f"Run:<br>{listed}")
    return True


def colour_key():
    """The legend for the whole app, so a colour never has to be guessed."""
    entries = [(LSMC, "LSMC / this project's estimate"),
               (BENCHMARK, "CRR binomial benchmark"),
               (MARKET, "observed market"),
               (REFERENCE, "reference lines")]
    html = " ".join(
        f'<span class="omc-badge"><span class="omc-swatch" '
        f'style="background:{colour}"></span>{label}</span>'
        for colour, label in entries)
    st.markdown(html, unsafe_allow_html=True)


def footer(test_count=None):
    extra = f" · {test_count} automated tests" if test_count else ""
    st.markdown(
        f'<div class="omc-footer">OptionMC Advanced Risk Lab · CSE402 '
        f'Numerical Analysis, Simulation and Modeling{extra}<br>'
        f'An educational numerical-analysis project. Not financial advice, and '
        f'not a recommendation to trade.</div>',
        unsafe_allow_html=True)


def planned_sections(sections, phase):
    """Outline of what a page will contain, while it is still a skeleton.

    Shown instead of an empty page so a reader can see the structure and the
    order of the argument before the content lands.
    """
    st.markdown(f"#### Sections on this page")
    for title, description in sections:
        st.markdown(f"**{title}** — {description}")
    st.write("")
    callout(f"This page's structure is in place; its content is built in "
            f"<b>{phase}</b>. The data it will read is already on disk, and "
            f"the panel below says how much of it.")


def data_readiness(availability_frame, needed):
    """A compact present/missing list for the tables a page depends on."""
    if availability_frame is None:
        return
    frame = availability_frame[availability_frame["item"].isin(needed)]
    if frame.empty:
        return
    ready = int(frame["present"].sum())
    with st.expander(f"Data this page needs: {ready} of {len(frame)} ready",
                     expanded=False):
        for _, row in frame.iterrows():
            mark = "ready" if row["present"] else "missing"
            kind = "" if row["present"] else " — run `python " + row["produced_by"] + "`"
            st.markdown(f"- **{row['item']}**: {mark}{kind}")


def pipeline(stages, title=None):
    """The chain from the base paper to the decision, as connected steps.

    Drawn in plain HTML rather than as a chart: it is a narrative of what
    happens in what order, not a plot of anything, and a diagram library would
    add a dependency for something a row of boxes says better.

    `stages` is a sequence of (label, detail) pairs.
    """
    if title:
        st.markdown(f"#### {title}")

    blocks = []
    for index, (label, detail) in enumerate(stages):
        blocks.append(
            f'<div class="omc-step"><div class="omc-step-n">{index + 1}</div>'
            f'<div class="omc-step-body"><b>{label}</b>'
            f'<div class="omc-step-detail">{detail}</div></div></div>')
        if index < len(stages) - 1:
            blocks.append('<div class="omc-step-arrow">&#8595;</div>')
    st.markdown(f'<div class="omc-pipeline">{"".join(blocks)}</div>',
                unsafe_allow_html=True)


def check_list(checks, title=None):
    """Pass/fail indicators, each traceable to the file it was checked against.

    Never colour alone: every row carries a word as well as a mark, because a
    projector washes out green and red into the same grey.
    """
    if title:
        st.markdown(f"#### {title}")

    passed = sum(1 for check in checks if check["passed"])
    kind = "good" if passed == len(checks) else "bad"
    callout(f"<b>{passed} of {len(checks)} invariants hold.</b> Each one is "
            f"re-derived from the tables currently on disk, so it is about "
            f"the results in front of you rather than a verdict recorded "
            f"earlier.", kind=kind)

    rows = []
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        colour = GOOD if check["passed"] else BAD
        rows.append(
            f'<div class="omc-check">'
            f'<span class="omc-check-mark" style="color:{colour}">{mark}</span>'
            f'<span class="omc-check-body"><b>{check["name"]}</b>'
            f'<div class="omc-check-detail">{check["detail"]} '
            f'<span class="omc-check-source">{check["source"]}</span></div>'
            f'</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
