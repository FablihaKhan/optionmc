"""Interactive Plotly charts for the dashboard.

Additional to the matplotlib figures, never a replacement: the PNG and PDF set
under `results/figures/` remains what goes in the report, and these are the
version a viewer can hover over.

They share the report's palette, imported from `src/plots.py`, so the same
quantity is the same colour in both. A chart the teacher sees on screen and a
chart in the printed report must not disagree about which line is the LSMC.

Deliberately absent: 3D charts, pie charts, rainbow scales and any second
y-axis. A dual-axis chart invents a relationship between two series that the
data does not contain, and it is the single most common way a finance
dashboard misleads.
"""
import plotly.graph_objects as go

from src.plots import BAND, BENCHMARK, LSMC, MARKET, ORDINAL_3, REFERENCE

NAVY = "#12233a"
INK_SOFT = "#4a5b70"
GRID = "#e3e8ef"

TEMPLATE = "omc"

# Ordered pairs -- 95% then 99% -- get one hue light to dark, because the
# levels are ranked rather than merely different.
LEVEL_RAMP = (ORDINAL_3[0], ORDINAL_3[2])


def register_template():
    """A Plotly template that matches the Streamlit theme."""
    import plotly.io as pio

    if TEMPLATE in pio.templates:
        return TEMPLATE

    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=14,
                  color=NAVY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        colorway=[LSMC, BENCHMARK, MARKET, REFERENCE],
        margin=dict(l=70, r=30, t=60, b=60),
        hoverlabel=dict(font_size=13, bgcolor="#ffffff",
                        bordercolor=GRID, font_color=NAVY),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID, zeroline=False, showline=True,
                   linecolor=GRID, ticks="outside", tickcolor=GRID),
        yaxis=dict(gridcolor=GRID, zeroline=False, showline=True,
                   linecolor=GRID, ticks="outside", tickcolor=GRID),
        title=dict(font=dict(size=17, color=NAVY), x=0.0, xanchor="left"),
    )
    pio.templates[TEMPLATE] = template
    return TEMPLATE


def finish(fig, title=None, xlabel=None, ylabel=None, height=430):
    """Apply the template and the axis labels every chart must carry."""
    register_template()
    fig.update_layout(template=TEMPLATE, height=height,
                      title=title or None,
                      xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def line(series, xlabel, ylabel, title=None, hover=None, height=430):
    """One or more named lines.

    `series` maps a label to (x, y) or (x, y, colour).
    """
    fig = go.Figure()
    for label, values in series.items():
        x, y, *rest = values
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label, mode="lines+markers",
            line=dict(width=2.5, color=rest[0] if rest else None),
            marker=dict(size=8),
            hovertemplate=hover or f"{label}<br>%{{x}}<br>%{{y}}<extra></extra>"))
    return finish(fig, title, xlabel, ylabel, height)


def scatter_with_identity(x, y_by_series, xlabel, ylabel, title=None,
                          hover_text=None, height=470):
    """A prediction-against-truth scatter with the y = x reference drawn in."""
    fig = go.Figure()
    lo = min([min(x)] + [min(v) for v in y_by_series.values()])
    hi = max([max(x)] + [max(v) for v in y_by_series.values()])
    pad = 0.04 * (hi - lo)

    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
        name="perfect prediction (y = x)",
        line=dict(color=REFERENCE, width=1.6, dash="dash"),
        hoverinfo="skip"))

    symbols = ("circle", "diamond", "square")
    for index, (label, values) in enumerate(y_by_series.items()):
        fig.add_trace(go.Scatter(
            x=x, y=values, name=label, mode="markers",
            marker=dict(size=11, symbol=symbols[index % len(symbols)],
                        line=dict(width=1.5, color="#ffffff")),
            text=hover_text,
            hovertemplate=(f"<b>{label}</b><br>%{{text}}<br>"
                           f"market %{{x:.4f}}<br>model %{{y:.4f}}<extra></extra>"
                           if hover_text is not None else None)))
    return finish(fig, title, xlabel, ylabel, height)


def grouped_bars(categories, series, xlabel, ylabel, title=None,
                 value_format="{:,.0f}", height=440):
    """One group per category, one bar per named series."""
    fig = go.Figure()
    colours = [LSMC, BENCHMARK, MARKET]
    for index, (label, values) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=list(categories), y=list(values), name=label,
            marker_color=colours[index % len(colours)],
            text=[value_format.format(v) for v in values],
            textposition="outside",
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:,.2f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
    return finish(fig, title, xlabel, ylabel, height)


def histogram_overlay(distributions, xlabel, title=None, bins=110,
                      vlines=None, height=450, log_y=False):
    """Two loss or return distributions drawn over one another.

    Outlined and translucent so neither hides the other, which a solid
    histogram inevitably does to whichever series is drawn second.
    """
    fig = go.Figure()
    colours = [LSMC, BENCHMARK, MARKET]
    for index, (label, values) in enumerate(distributions.items()):
        fig.add_trace(go.Histogram(
            x=values, name=label, nbinsx=bins, opacity=0.55,
            marker=dict(color=colours[index % len(colours)],
                        line=dict(width=1, color=colours[index % len(colours)])),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y}} scenarios<extra></extra>"))

    for label, value, colour in (vlines or []):
        fig.add_vline(x=value, line=dict(color=colour, width=2, dash="dash"),
                      annotation_text=label, annotation_position="top",
                      annotation_font_size=12)

    fig.update_layout(barmode="overlay")
    if log_y:
        fig.update_yaxes(type="log")
    return finish(fig, title, xlabel, "number of scenarios", height)


def frontier(x, y, labels, pareto=None, highlight=None, xlabel="", ylabel="",
             title=None, hover=None, height=520):
    """The protection-cost frontier: one point per real listed contract.

    Up is more tail protection, right is more expensive. A candidate that some
    other candidate beats on both counts is drawn hollow and left off the
    connecting line -- a dominated hedge is never presented as a choice.

    `hover` is an optional DataFrame whose columns become the tooltip, so the
    chart can carry the strike, the quotes, the premium and every reduction
    without any of it cluttering the plot itself.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pareto = (np.ones(x.shape, dtype=bool) if pareto is None
              else np.asarray(pareto, dtype=bool))
    text = [str(label) for label in labels]

    if hover is not None:
        columns = list(hover.columns)
        customdata = hover.to_numpy()
        detail = "<br>".join(
            f"{name}: %{{customdata[{index}]}}"
            for index, name in enumerate(columns))
        template = "<b>%{text}</b><br>" + detail + "<extra></extra>"
    else:
        customdata = None
        template = ("<b>%{text}</b><br>cost %{x:.2f}%<br>"
                    "reduction %{y:.2f}%<extra></extra>")

    def subset(mask):
        return (None if customdata is None else customdata[mask])

    fig = go.Figure()
    order = np.argsort(x)
    front = order[pareto[order]]
    if front.size > 1:
        fig.add_trace(go.Scatter(
            x=x[front], y=y[front], mode="lines", name="Pareto frontier",
            line=dict(color=LSMC, width=2.5), hoverinfo="skip"))

    if pareto.any():
        fig.add_trace(go.Scatter(
            x=x[pareto], y=y[pareto], mode="markers+text",
            name="Pareto efficient",
            text=[t for t, keep in zip(text, pareto) if keep],
            textposition="top center", textfont=dict(size=12, color=INK_SOFT),
            marker=dict(size=16, color=LSMC,
                        line=dict(width=2, color="#ffffff")),
            customdata=subset(pareto), hovertemplate=template))
    if (~pareto).any():
        fig.add_trace(go.Scatter(
            x=x[~pareto], y=y[~pareto], mode="markers+text", name="dominated",
            text=[t for t, keep in zip(text, ~pareto) if keep],
            textposition="top center", textfont=dict(size=12, color=INK_SOFT),
            marker=dict(size=16, color="#ffffff",
                        line=dict(width=2, color=REFERENCE)),
            customdata=subset(~pareto), hovertemplate=template))

    if highlight is not None:
        fig.add_trace(go.Scatter(
            x=[x[highlight]], y=[y[highlight]], mode="markers",
            name="best for the selected objective",
            marker=dict(size=32, color="rgba(0,0,0,0)",
                        line=dict(width=3, color=BENCHMARK)),
            hoverinfo="skip"))

    span = float(y.max() - y.min()) or 1.0
    fig.update_yaxes(range=[y.min() - 0.18 * span, y.max() + 0.28 * span],
                     ticksuffix="%")
    fig.update_xaxes(ticksuffix="%")
    return finish(fig, title, xlabel, ylabel, height)


def efficiency_by_strike(strikes, series, xlabel="strike [$]",
                         ylabel="CVaR avoided per $1 of premium",
                         breakeven=1.0, height=430):
    """How many dollars of tail loss each dollar of premium buys.

    The break-even line is drawn because the number is meaningless without it:
    below one, the hedge costs more than the tail loss it removes.
    """
    fig = go.Figure()
    for index, (label, values) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=[f"{s:g}" for s in strikes], y=list(values), name=label,
            marker_color=LEVEL_RAMP[index % len(LEVEL_RAMP)],
            text=[f"${v:,.2f}" for v in values], textposition="outside",
            hovertemplate=f"<b>{label}</b><br>strike %{{x}}<br>"
                          f"%{{y:$,.2f}} per $1<extra></extra>"))
    if breakeven is not None:
        fig.add_hline(y=breakeven, line=dict(color=REFERENCE, width=1.6,
                                             dash="dash"),
                      annotation_text="break-even: $1 saved per $1 spent",
                      annotation_position="top left", annotation_font_size=11)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    fig.update_yaxes(tickprefix="$")
    return finish(fig, None, xlabel, ylabel, height)


def reduction_by_strike(moneyness, series, height=430):
    """Each risk measure's reduction as the hedge moves toward the money.

    VaR is dashed and CVaR solid, so the two measures stay apart for a reader
    who cannot separate the two blues.
    """
    import numpy as np

    x = np.asarray(moneyness, dtype=float) * 100.0
    styles = {
        "95% VaR": (LEVEL_RAMP[0], "dash"),
        "95% CVaR": (LEVEL_RAMP[0], "solid"),
        "99% VaR": (LEVEL_RAMP[1], "dash"),
        "99% CVaR": (LEVEL_RAMP[1], "solid"),
    }
    fig = go.Figure()
    for label, values in series.items():
        colour, dash = styles.get(label, (MARKET, "solid"))
        fig.add_trace(go.Scatter(
            x=x, y=values, mode="lines+markers", name=label,
            line=dict(color=colour, width=2.6, dash=dash),
            marker=dict(size=9),
            hovertemplate=f"<b>{label}</b><br>strike %{{x:.1f}}% of spot<br>"
                          f"%{{y:.2f}}% reduction<extra></extra>"))
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")
    return finish(fig, None, "strike as a percentage of spot",
                  "reduction against the unhedged portfolio", height)


def band_line(x, y, lower, upper, label, xlabel, ylabel, title=None,
              height=430, band_label="uncertainty band"):
    """A line with a shaded interval, for anything carrying a standard error."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1], y=list(upper) + list(lower)[::-1],
        fill="toself", fillcolor=BAND, opacity=0.45, line=dict(width=0),
        name=band_label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=label,
        line=dict(color=LSMC, width=2.5), marker=dict(size=8),
        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    return finish(fig, title, xlabel, ylabel, height)


def loss_story(unhedged_losses, protected_losses, grid_spots, grid_prices,
               strike, spot, var_unhedged=None, var_protected=None,
               bins=90, height=340):
    """The project's argument in one row: the risk, the instrument, the result.

    Three panels rather than one overlay. The two loss distributions share an
    x-axis so the truncated left tail is a visible fact rather than a claim,
    and the middle panel shows the payoff that does the truncating -- without
    it a reader sees that the tail got shorter but not why.
    """
    import numpy as np
    from plotly.subplots import make_subplots

    unhedged_losses = np.asarray(unhedged_losses, dtype=float)
    protected_losses = np.asarray(protected_losses, dtype=float)

    lo = float(min(unhedged_losses.min(), protected_losses.min()))
    hi = float(max(unhedged_losses.max(), protected_losses.max()))
    edges = np.linspace(lo, hi, bins + 1)

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.07,
        subplot_titles=("1 · The risk: 100 SPY shares",
                        "2 · The instrument: the put at the horizon",
                        "3 · The result: shares plus put"))

    for column, (losses, colour, name) in enumerate(
            ((unhedged_losses, LSMC, "SPY only"),
             (protected_losses, BENCHMARK, "SPY + put")), start=1):
        target = 1 if column == 1 else 3
        counts, _ = np.histogram(losses, bins=edges)
        fig.add_trace(go.Bar(
            x=0.5 * (edges[:-1] + edges[1:]), y=counts, name=name,
            marker=dict(color=colour, line=dict(width=0)),
            width=(edges[1] - edges[0]),
            hovertemplate=f"<b>{name}</b><br>loss %{{x:$,.0f}}<br>"
                          f"%{{y:,}} scenarios<extra></extra>"),
            row=1, col=target)

    # The put's horizon value against the spot: the payoff panel.
    fig.add_trace(go.Scatter(
        x=grid_spots, y=grid_prices, mode="lines", name="put value",
        line=dict(color=BENCHMARK, width=3),
        hovertemplate="SPY %{x:$,.0f}<br>put %{y:$,.2f}<extra></extra>"),
        row=1, col=2)
    # The strike and the spot are only a few dollars apart, so their labels
    # collide unless they are pushed to opposite corners.
    fig.add_vline(x=strike, line=dict(color=REFERENCE, width=1.4, dash="dash"),
                  annotation_text=f"strike {strike:g}", annotation_font_size=11,
                  annotation_position="top left", row=1, col=2)
    fig.add_vline(x=spot, line=dict(color=MARKET, width=1.4),
                  annotation_text="spot today", annotation_font_size=11,
                  annotation_position="bottom right", row=1, col=2)

    for column, value, colour in ((1, var_unhedged, LSMC),
                                  (3, var_protected, BENCHMARK)):
        if value is not None:
            fig.add_vline(x=value, line=dict(color=REFERENCE, width=1.6,
                                             dash="dash"),
                          annotation_text=f"99% VaR ${value:,.0f}",
                          annotation_font_size=11, row=1, col=column)

    for column in (1, 3):
        fig.update_xaxes(range=[lo, hi], title_text="loss over 10 days [$]",
                         tickprefix="$", row=1, col=column)
        fig.update_yaxes(type="log", title_text="scenarios" if column == 1 else "",
                         row=1, col=column)
    fig.update_xaxes(title_text="SPY at the horizon [$]", tickprefix="$",
                     row=1, col=2)
    fig.update_yaxes(title_text="put value [$]", tickprefix="$", row=1, col=2)

    register_template()
    fig.update_layout(template=TEMPLATE, height=height, showlegend=False,
                      bargap=0, margin=dict(l=60, r=25, t=52, b=50))
    fig.update_annotations(font_size=13)
    return fig


def paths_chart(times, paths, strike=None, spot=None, n_show=120, height=420):
    """A sample of the simulated price paths.

    Drawn as one trace with gaps between paths rather than one trace each: a
    hundred separate traces makes the legend meaningless and the browser slow,
    and none of the individual paths is a series anyone needs to identify.
    """
    import numpy as np

    paths = np.asarray(paths, dtype=float)
    times = np.asarray(times, dtype=float)
    n_show = int(min(n_show, paths.shape[0]))

    xs, ys = [], []
    for row in paths[:n_show]:
        xs.extend(times.tolist() + [None])
        ys.extend(row.tolist() + [None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name=f"{n_show} simulated paths",
        line=dict(color=BAND, width=1), opacity=0.75, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=times, y=paths.mean(axis=0), mode="lines", name="mean path",
        line=dict(color=LSMC, width=3),
        hovertemplate="t %{x:.3f} yr<br>mean %{y:$,.2f}<extra></extra>"))

    if strike is not None:
        fig.add_hline(y=strike, line=dict(color=REFERENCE, width=1.6,
                                          dash="dash"),
                      annotation_text=f"strike {strike:g}",
                      annotation_position="top left",
                      annotation_font_size=11)
    if spot is not None:
        # Rarely worth drawing: every path starts at the spot, so the line sits
        # under the mean path and its label collides with the strike's.
        fig.add_hline(y=spot, line=dict(color=MARKET, width=1.4),
                      annotation_text="spot today", annotation_font_size=11,
                      annotation_position="bottom right")

    fig.update_yaxes(tickprefix="$")
    return finish(fig, None, "time [years]", "price [$]", height)


def price_vs_spot(frame, strike, spot=None, height=430):
    """LSMC and the lattice across a range of spots, with intrinsic value.

    Intrinsic is drawn because it is the floor an American put cannot go below;
    a curve dipping under it would be a bug, and here it is visible.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["spot"], y=frame["intrinsic"], mode="lines",
        name="intrinsic value", line=dict(color=REFERENCE, width=1.6,
                                          dash="dot"),
        hovertemplate="SPY %{x:$,.0f}<br>intrinsic %{y:$,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=frame["spot"], y=frame["binomial"], mode="lines",
        name="CRR binomial", line=dict(color=BENCHMARK, width=3),
        hovertemplate="SPY %{x:$,.0f}<br>CRR %{y:$,.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=frame["spot"], y=frame["lsmc"], mode="lines+markers", name="LSMC",
        line=dict(color=LSMC, width=2.2, dash="dash"), marker=dict(size=7),
        hovertemplate="SPY %{x:$,.0f}<br>LSMC %{y:$,.4f}<extra></extra>"))

    fig.add_vline(x=strike, line=dict(color=REFERENCE, width=1.4, dash="dash"),
                  annotation_text=f"strike {strike:g}",
                  annotation_position="top left", annotation_font_size=11)
    if spot is not None:
        fig.add_vline(x=spot, line=dict(color=MARKET, width=1.4),
                      annotation_text="spot today",
                      annotation_position="bottom right",
                      annotation_font_size=11)

    fig.update_xaxes(tickprefix="$")
    fig.update_yaxes(tickprefix="$")
    return finish(fig, None, "spot at valuation [$]", "put value [$]", height)


def exercise_boundary_chart(time_remaining, boundary, strike, height=420):
    """The estimated early-exercise boundary, against time left on the option.

    Read right to left as the option ages. Nodes where too few paths were in
    the money to fit a regression have no boundary and are simply absent, which
    is honest -- the alternative is drawing a line through nothing.
    """
    import numpy as np

    time_remaining = np.asarray(time_remaining, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    finite = np.isfinite(boundary)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_remaining[finite], y=boundary[finite], mode="lines+markers",
        name="exercise below this price", line=dict(color=LSMC, width=2.5),
        marker=dict(size=6),
        hovertemplate="%{x:.3f} yr left<br>exercise below %{y:$,.2f}"
                      "<extra></extra>"))
    fig.add_hline(y=strike, line=dict(color=REFERENCE, width=1.6, dash="dash"),
                  annotation_text=f"strike {strike:g}", annotation_font_size=11)

    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(tickprefix="$")
    return finish(fig, None, "time left on the option [years]",
                  "spot at which exercise beats waiting [$]", height)


def convergence_chart(frame, height=420):
    """Price against the number of paths, with the lattice as the target."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(frame["n_paths"]) + list(frame["n_paths"])[::-1],
        y=list(frame["upper"]) + list(frame["lower"])[::-1],
        fill="toself", fillcolor=BAND, opacity=0.45, line=dict(width=0),
        name="LSMC ± 2 standard errors", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=frame["n_paths"], y=frame["benchmark"], mode="lines",
        name="CRR binomial", line=dict(color=BENCHMARK, width=2.2,
                                       dash="dash"),
        hovertemplate="CRR %{y:$,.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=frame["n_paths"], y=frame["price"], mode="lines+markers",
        name="LSMC price", line=dict(color=LSMC, width=2.6),
        marker=dict(size=9),
        hovertemplate="%{x:,} paths<br>LSMC %{y:$,.4f}<extra></extra>"))

    fig.update_xaxes(type="log", tickvals=list(frame["n_paths"]),
                     ticktext=[f"{int(n):,}" for n in frame["n_paths"]])
    fig.update_yaxes(tickprefix="$")
    return finish(fig, None, "number of simulated paths", "put price [$]",
                  height)


def calibration_split_chart(strikes, roles, expiries, spot, height=None):
    """Where the calibration and held-out strikes sit, expiry by expiry.

    The point of this chart is that the two sets never coincide. That is easier
    to trust when it is visible than when it is asserted in a caption.
    """
    import numpy as np

    strikes = np.asarray(strikes, dtype=float)
    roles = np.asarray(roles)
    expiries = np.asarray(expiries)
    labels = list(dict.fromkeys(expiries))

    fig = go.Figure()
    for row, expiry in enumerate(labels):
        here = expiries == expiry
        for role, colour, symbol, filled in (
                ("calibration", LSMC, "circle", True),
                ("test", BENCHMARK, "diamond", False)):
            mask = here & (roles == role)
            if not mask.any():
                continue
            name = ("used to fit the curve" if role == "calibration"
                    else "held out")
            fig.add_trace(go.Scatter(
                x=strikes[mask], y=np.full(int(mask.sum()), row),
                mode="markers", name=name, showlegend=(row == 0),
                marker=dict(size=14, symbol=symbol,
                            color=colour if filled else "#ffffff",
                            line=dict(width=2,
                                      color="#ffffff" if filled else colour)),
                hovertemplate=f"<b>{name}</b><br>strike %{{x:$,.0f}}"
                              f"<br>{expiry}<extra></extra>"))

    fig.add_vline(x=spot, line=dict(color=REFERENCE, width=1.6, dash="dash"),
                  annotation_text=f"spot ${spot:,.2f}",
                  annotation_position="top right", annotation_font_size=11)

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(labels))),
                     ticktext=labels, showgrid=False,
                     range=[-0.6, len(labels) - 0.4])
    fig.update_xaxes(tickprefix="$")
    finish(fig, None, "strike [$]", "expiry",
           height or (170 + 80 * len(labels)))
    # The tick labels are full dates, which the default left margin clips.
    fig.update_layout(margin=dict(l=130, r=30, t=60, b=60))
    return fig


def smile_chart(calibration_x, calibration_y, curve_x, curve_y, test_x, test_y,
                height=440):
    """The fitted smile, the quotes that made it, and where it was queried.

    `test_y` is the volatility each held-out contract was *given* -- a value
    read off the curve, not a market observation. The held-out contracts' own
    implied volatilities are deliberately absent: drawing them beside the curve
    would suggest they helped shape it, which is the one thing this validation
    is built to avoid.
    """
    import numpy as np

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_x, y=np.asarray(curve_y, dtype=float) * 100.0, mode="lines",
        name="PCHIP smile, fitted to calibration points only",
        line=dict(color=LSMC, width=3),
        hovertemplate="log-moneyness %{x:.4f}<br>%{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=calibration_x, y=np.asarray(calibration_y, dtype=float) * 100.0,
        mode="markers", name="calibration quotes",
        marker=dict(size=13, color=LSMC, line=dict(width=2, color="#ffffff")),
        hovertemplate="log-moneyness %{x:.4f}<br>implied %{y:.2f}%"
                      "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=test_x, y=np.asarray(test_y, dtype=float) * 100.0, mode="markers",
        name="held out: volatility read off the curve",
        marker=dict(size=13, symbol="diamond", color="#ffffff",
                    line=dict(width=2.5, color=BENCHMARK)),
        hovertemplate="log-moneyness %{x:.4f}<br>given %{y:.2f}%"
                      "<extra></extra>"))

    fig.add_vline(x=0.0, line=dict(color=REFERENCE, width=1.4),
                  annotation_text="at the money",
                  annotation_position="bottom right", annotation_font_size=11)
    fig.update_yaxes(ticksuffix="%")
    return finish(fig, None, "log-moneyness  log(K / spot)",
                  "American implied volatility", height)


def error_chart(x, errors_by_model, xlabel, band=None, hover_text=None,
                x_prefix="", height=420):
    """Prediction error against any contract characteristic.

    The zero line is the reference, and the optional band is the LSMC's own
    two-standard-error range -- errors inside it are the simulation's sampling
    noise rather than a disagreement with the market.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    fig = go.Figure()

    if band is not None:
        width = 2.0 * np.asarray(band, dtype=float)[order]
        fig.add_trace(go.Scatter(
            x=list(x[order]) + list(x[order])[::-1],
            y=list(width) + list(-width)[::-1], fill="toself",
            fillcolor=BAND, opacity=0.45, line=dict(width=0),
            name="LSMC plus or minus 2 standard errors", hoverinfo="skip"))

    fig.add_hline(y=0.0, line=dict(color=REFERENCE, width=1.6, dash="dash"))

    colours = {"CRR binomial": BENCHMARK, "LSMC": LSMC}
    symbols = {"CRR binomial": "circle", "LSMC": "diamond"}
    for label, values in errors_by_model.items():
        values = np.asarray(values, dtype=float)
        fig.add_trace(go.Scatter(
            x=x[order], y=values[order], mode="lines+markers", name=label,
            line=dict(color=colours.get(label, MARKET), width=2.2),
            marker=dict(size=9, symbol=symbols.get(label, "circle")),
            text=(np.asarray(hover_text)[order]
                  if hover_text is not None else None),
            hovertemplate=(f"<b>{label}</b><br>%{{text}}<br>"
                           f"error %{{y:$,.4f}}<extra></extra>"
                           if hover_text is not None else
                           f"<b>{label}</b><br>%{{x}}<br>"
                           f"error %{{y:$,.4f}}<extra></extra>")))

    if x_prefix:
        fig.update_xaxes(tickprefix=x_prefix)
    fig.update_yaxes(tickprefix="$")
    return finish(fig, None, xlabel, "model minus market [$]", height)


def error_heatmap(pivot, height=None):
    """Mean absolute error by expiry and moneyness bucket.

    A single-hue ramp, because the value is a magnitude rather than a category,
    and every cell carries its number so the figure never depends on colour.
    """
    import numpy as np

    values = pivot.to_numpy(dtype=float)
    text = [[("--" if not np.isfinite(v) else f"${v:.3f}") for v in row]
            for row in values]

    fig = go.Figure(go.Heatmap(
        z=values, x=[str(column) for column in pivot.columns],
        y=[str(index) for index in pivot.index],
        colorscale="Blues", text=text, texttemplate="%{text}",
        textfont=dict(size=13),
        colorbar=dict(title="mean abs error", tickprefix="$"),
        hovertemplate="%{y}<br>%{x}<br>%{z:$,.4f}<extra></extra>"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return finish(fig, None, "moneyness bucket", "expiry",
                  height or (170 + 90 * pivot.shape[0]))


def spacing_study_chart(spacings, series, noise_floor=None, height=420):
    """Held-out accuracy as the calibration grid is thinned."""
    fig = go.Figure()
    colours = {"CRR binomial": BENCHMARK, "LSMC": LSMC}
    for label, values in series.items():
        fig.add_trace(go.Scatter(
            x=spacings, y=values, mode="lines+markers", name=label,
            line=dict(color=colours.get(label, MARKET), width=2.6),
            marker=dict(size=10),
            hovertemplate=f"<b>{label}</b><br>$%{{x}} spacing<br>"
                          f"MAE %{{y:$,.4f}}<extra></extra>"))
    if noise_floor is not None:
        fig.add_hline(y=noise_floor, line=dict(color=REFERENCE, width=1.5,
                                               dash="dot"),
                      annotation_text="LSMC Monte Carlo noise floor",
                      annotation_position="top left", annotation_font_size=11)

    fig.update_xaxes(type="log", tickvals=list(spacings),
                     ticktext=[f"${s:g}" for s in spacings])
    fig.update_yaxes(rangemode="tozero", tickprefix="$")
    return finish(fig, None, "spacing between calibration strikes",
                  "mean absolute error on held-out contracts", height)


def loss_distribution(unhedged, protected, label_a="SPY only",
                      label_b="SPY + put", var_lines=(), bins=120,
                      height=460):
    """Two loss distributions, with the value-at-risk thresholds marked.

    Drawn on a log count axis. The tail is where a hedge earns its premium and
    it holds a few dozen scenarios out of fifty thousand; on a linear axis it
    is invisible, which is the opposite of what this chart is for.

    `var_lines` is a sequence of (label, value, colour).
    """
    import numpy as np

    unhedged = np.asarray(unhedged, dtype=float)
    protected = np.asarray(protected, dtype=float)
    edges = np.histogram_bin_edges(np.concatenate([unhedged, protected]), bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    fig = go.Figure()
    for values, colour, name in ((unhedged, LSMC, label_a),
                                 (protected, BENCHMARK, label_b)):
        counts, _ = np.histogram(values, bins=edges)
        fig.add_trace(go.Bar(
            x=centres, y=counts, name=name, width=width, opacity=0.62,
            marker=dict(color=colour, line=dict(width=0)),
            hovertemplate=f"<b>{name}</b><br>loss %{{x:$,.0f}}<br>"
                          f"%{{y:,}} scenarios<extra></extra>"))

    # Two thresholds a few hundred dollars apart put their labels on top of
    # each other, so successive lines are staggered downwards.
    for index, (label, value, colour) in enumerate(var_lines):
        fig.add_vline(x=value, line=dict(color=colour, width=2, dash="dash"),
                      annotation_text=f"{label} ${value:,.0f}",
                      annotation_font_size=11,
                      annotation_position="top right",
                      annotation_yshift=-22 * index)

    fig.add_vline(x=0.0, line=dict(color=REFERENCE, width=1.2))
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_xaxes(tickprefix="$")
    fig.update_yaxes(type="log")
    return finish(fig, None, "loss over the horizon [$]",
                  "scenarios (log scale)", height)


def return_distribution(returns, mean, std, bins=90, height=440,
                        label="observed daily returns"):
    """Observed returns against the normal that GBM assumes they follow.

    The y axis is a log density: in the middle the two agree and the comparison
    is uninteresting, and the whole point sits in the tails, where a linear
    axis shows nothing.
    """
    import numpy as np

    returns = np.asarray(returns, dtype=float) * 100.0
    mean, std = mean * 100.0, std * 100.0
    counts, edges = np.histogram(returns, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    grid = np.linspace(returns.min() * 1.05, returns.max() * 1.05, 500)
    normal = (np.exp(-0.5 * ((grid - mean) / std) ** 2)
              / (std * np.sqrt(2 * np.pi)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres, y=counts, name=label, width=(edges[1] - edges[0]),
        marker=dict(color=MARKET, line=dict(width=0)), opacity=0.6,
        hovertemplate="return %{x:.2f}%<br>density %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=grid, y=normal, mode="lines", name="normal assumed by GBM",
        line=dict(color=LSMC, width=3),
        hovertemplate="return %{x:.2f}%<br>density %{y:.4f}<extra></extra>"))

    positive = counts[counts > 0]
    if positive.size:
        fig.update_yaxes(type="log",
                         range=[np.log10(positive.min() * 0.6),
                                np.log10(max(counts.max(), normal.max()) * 1.6)])
    fig.update_xaxes(ticksuffix="%")
    return finish(fig, None, "daily log return", "density (log scale)", height)


def stress_chart(shocks, stock_loss_percent, protected_loss_percent,
                 put_values, height=560):
    """What a crash does to each portfolio, and the payoff that limits it.

    Two panels rather than two y-axes on one figure. Loss percentage and option
    value are different quantities on different scales, and overlaying them
    would invent a relationship the data does not contain.
    """
    import numpy as np
    from plotly.subplots import make_subplots

    shocks = np.asarray(shocks, dtype=float) * 100.0
    order = np.argsort(shocks)
    shocks = shocks[order]
    stock = np.asarray(stock_loss_percent, dtype=float)[order]
    protected = np.asarray(protected_loss_percent, dtype=float)[order]
    puts = np.asarray(put_values, dtype=float)[order]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.09, row_heights=[0.62, 0.38],
                        subplot_titles=("Loss, measured from each portfolio's "
                                        "own starting value",
                                        "What the put is worth at the horizon"))

    fig.add_trace(go.Scatter(
        x=shocks, y=stock, mode="lines+markers", name="SPY only",
        line=dict(color=LSMC, width=3), marker=dict(size=11),
        hovertemplate="shock %{x:.0f}%<br>loss %{y:.2f}%<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=shocks, y=protected, mode="lines+markers", name="SPY + put",
        line=dict(color=BENCHMARK, width=3, dash="dash"),
        marker=dict(size=11, symbol="square"), fill="tonexty",
        fillcolor="rgba(235,104,52,0.12)",
        hovertemplate="shock %{x:.0f}%<br>loss %{y:.2f}%<extra></extra>"),
        row=1, col=1)

    for x, a, b in zip(shocks, stock, protected):
        if a - b > 0.4:
            fig.add_annotation(x=x, y=(a + b) / 2, text=f"{a - b:.1f} pts",
                               showarrow=False, xshift=34, font_size=11,
                               font_color=INK_SOFT, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=shocks, y=puts, mode="lines+markers+text", name="put value",
        line=dict(color=BENCHMARK, width=3), marker=dict(size=11,
                                                         symbol="square"),
        text=[f"${v:,.0f}" for v in puts], textposition="top center",
        textfont=dict(size=11, color=INK_SOFT), showlegend=False,
        hovertemplate="shock %{x:.0f}%<br>put %{y:$,.2f} per share"
                      "<extra></extra>"),
        row=2, col=1)

    register_template()
    fig.update_layout(template=TEMPLATE, height=height,
                      margin=dict(l=70, r=40, t=70, b=60))
    fig.update_xaxes(ticksuffix="%", row=2, col=1,
                     title_text="shock to SPY over the horizon")
    fig.update_yaxes(ticksuffix="%", row=1, col=1, title_text="loss")
    # Headroom for the value labels, which sit above their markers and would
    # otherwise be clipped by the subplot title.
    fig.update_yaxes(tickprefix="$", row=2, col=1,
                     title_text="value per share",
                     range=[0, float(puts.max()) * 1.22])
    fig.update_annotations(font_size=13)
    return fig


def quantile_chart(empirical, simulated, xlabel, ylabel, height=520):
    """Matched quantiles of two horizon distributions, against the diagonal."""
    import numpy as np

    empirical = np.asarray(empirical, dtype=float) * 100.0
    simulated = np.asarray(simulated, dtype=float) * 100.0
    lo = float(min(empirical.min(), simulated.min()))
    hi = float(max(empirical.max(), simulated.max()))
    pad = 0.05 * (hi - lo)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
        name="the two agree", line=dict(color=REFERENCE, width=1.6,
                                        dash="dash"),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=simulated, y=empirical, mode="markers", name="matched quantiles",
        marker=dict(size=8, color=MARKET, line=dict(width=1,
                                                    color="#ffffff")),
        hovertemplate="GBM %{x:.2f}%<br>bootstrap %{y:.2f}%<extra></extra>"))

    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%", scaleanchor="x", scaleratio=1)
    return finish(fig, None, xlabel, ylabel, height)


def convergence_with_order(sizes, values, lower, upper, benchmark, errors,
                           fitted_order=None, height=430):
    """Price and error against the number of paths, side by side.

    The left panel answers "is it settling", the right answers "at what rate".
    The right is log-log because a power law is a straight line there and
    nowhere else, which is what makes the fitted slope readable.
    """
    import numpy as np
    from plotly.subplots import make_subplots

    sizes = np.asarray(sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=("Price, with two standard errors",
                        "Error against the lattice, log-log"))

    fig.add_trace(go.Scatter(
        x=list(sizes) + list(sizes)[::-1],
        y=list(upper) + list(lower)[::-1], fill="toself", fillcolor=BAND,
        opacity=0.45, line=dict(width=0), name="+/- 2 standard errors",
        hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sizes, y=[benchmark] * len(sizes), mode="lines", name="CRR binomial",
        line=dict(color=BENCHMARK, width=2.2, dash="dash"),
        hovertemplate="CRR %{y:$,.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sizes, y=values, mode="lines+markers", name="LSMC mean price",
        line=dict(color=LSMC, width=2.6), marker=dict(size=10),
        hovertemplate="%{x:,} paths<br>%{y:$,.4f}<extra></extra>"),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sizes, y=errors, mode="lines+markers", name="root mean square error",
        line=dict(color=LSMC, width=2.6), marker=dict(size=10),
        showlegend=False,
        hovertemplate="%{x:,} paths<br>RMSE %{y:$,.4f}<extra></extra>"),
        row=1, col=2)

    if fitted_order is not None:
        reference = errors[0] * (sizes / sizes[0]) ** (-0.5)
        fig.add_trace(go.Scatter(
            x=sizes, y=reference, mode="lines",
            name="theoretical N^-0.5",
            line=dict(color=REFERENCE, width=2, dash="dot"),
            hovertemplate="theory %{y:$,.4f}<extra></extra>"), row=1, col=2)

    register_template()
    # The horizontal legend and the subplot titles both want the top of the
    # figure; the legend is lifted clear and the margin opened to fit both.
    fig.update_layout(template=TEMPLATE, height=height,
                      margin=dict(l=70, r=30, t=100, b=60),
                      legend=dict(y=1.16))
    for column in (1, 2):
        fig.update_xaxes(type="log", tickvals=list(sizes),
                         ticktext=[f"{int(n):,}" for n in sizes],
                         title_text="paths", row=1, col=column)
    fig.update_yaxes(tickprefix="$", title_text="price", row=1, col=1)
    fig.update_yaxes(type="log", tickprefix="$", title_text="RMSE",
                     row=1, col=2)
    fig.update_annotations(font_size=13)
    return fig


def discretisation_chart(steps, price, mc_error, discretisation_error,
                         runtime, bermudan, height=430):
    """What more exercise dates buy, and what they cost.

    The two error sources are separated because they move in opposite
    directions and can cancel: too few exercise dates under-prices the option,
    while Monte Carlo noise is symmetric. A single "total error" curve can look
    flat while both parts are large.
    """
    import numpy as np
    from plotly.subplots import make_subplots

    steps = np.asarray(steps, dtype=float)
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=("The two error sources, separated",
                        "What it costs to run"))

    fig.add_trace(go.Scatter(
        x=steps, y=np.abs(mc_error), mode="lines+markers",
        name="Monte Carlo error", line=dict(color=LSMC, width=2.6),
        marker=dict(size=10),
        hovertemplate="%{x:.0f} steps<br>%{y:$,.4f}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=np.abs(discretisation_error), mode="lines+markers",
        name="discretisation error", line=dict(color=BENCHMARK, width=2.6,
                                               dash="dash"),
        marker=dict(size=10, symbol="square"),
        hovertemplate="%{x:.0f} steps<br>%{y:$,.4f}<extra></extra>"),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=runtime, mode="lines+markers", name="runtime",
        line=dict(color=MARKET, width=2.6), marker=dict(size=10),
        showlegend=False,
        hovertemplate="%{x:.0f} steps<br>%{y:.3f} s<extra></extra>"),
        row=1, col=2)

    register_template()
    fig.update_layout(template=TEMPLATE, height=height,
                      margin=dict(l=70, r=30, t=100, b=60),
                      legend=dict(y=1.16))
    for column in (1, 2):
        fig.update_xaxes(title_text="exercise dates", tickvals=list(steps),
                         row=1, col=column)
    fig.update_yaxes(tickprefix="$", title_text="absolute error", row=1, col=1)
    fig.update_yaxes(ticksuffix=" s", title_text="seconds", row=1, col=2,
                     rangemode="tozero")
    fig.update_annotations(font_size=13)
    return fig


def regression_degree_chart(path_counts, series, benchmark, height=440):
    """Price against paths, one line per polynomial degree.

    The degrees are ordered, so they get one hue light to dark rather than
    three unrelated colours -- the ordering is part of what the reader needs.
    """
    import numpy as np

    ramp = ORDINAL_3 if len(series) <= 3 else ORDINAL_3 + [MARKET]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(path_counts), y=[benchmark] * len(path_counts), mode="lines",
        name="Bermudan benchmark",
        line=dict(color=REFERENCE, width=2, dash="dash"),
        hovertemplate="benchmark %{y:$,.4f}<extra></extra>"))
    for index, (label, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=list(path_counts), y=list(values), mode="lines+markers",
            name=label, line=dict(color=ramp[index % len(ramp)], width=2.6),
            marker=dict(size=10),
            hovertemplate=f"<b>{label}</b><br>%{{x:,}} paths<br>"
                          f"%{{y:$,.4f}}<extra></extra>"))

    fig.update_xaxes(type="log", tickvals=list(path_counts),
                     ticktext=[f"{int(n):,}" for n in path_counts])
    fig.update_yaxes(tickprefix="$")
    return finish(fig, None, "paths", "mean price over the replications",
                  height)


def grid_against_lattice(spots, lsmc, binomial, intrinsic, strike,
                         height=430):
    """The whole pricing grid against the lattice, and the gap underneath."""
    import numpy as np
    from plotly.subplots import make_subplots

    spots = np.asarray(spots, dtype=float)
    lsmc = np.asarray(lsmc, dtype=float)
    binomial = np.asarray(binomial, dtype=float)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.62, 0.38],
                        subplot_titles=("The grid and the lattice",
                                        "LSMC minus lattice"))

    fig.add_trace(go.Scatter(
        x=spots, y=intrinsic, mode="lines", name="intrinsic value",
        line=dict(color=REFERENCE, width=1.6, dash="dot"),
        hovertemplate="spot %{x:$,.0f}<br>%{y:$,.2f}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spots, y=binomial, mode="lines", name="CRR binomial",
        line=dict(color=BENCHMARK, width=3),
        hovertemplate="spot %{x:$,.0f}<br>%{y:$,.4f}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spots, y=lsmc, mode="lines+markers", name="LSMC grid",
        line=dict(color=LSMC, width=2.2, dash="dash"), marker=dict(size=6),
        hovertemplate="spot %{x:$,.0f}<br>%{y:$,.4f}<extra></extra>"),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=spots, y=lsmc - binomial, mode="lines+markers", name="difference",
        line=dict(color=LSMC, width=2), marker=dict(size=5),
        showlegend=False,
        hovertemplate="spot %{x:$,.0f}<br>%{y:$,.4f}<extra></extra>"),
        row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color=REFERENCE, width=1.4, dash="dash"),
                  row=2, col=1)
    fig.add_vline(x=strike, line=dict(color=REFERENCE, width=1.2, dash="dot"),
                  annotation_text=f"strike {strike:g}",
                  annotation_position="top left", annotation_font_size=11,
                  row=1, col=1)

    register_template()
    fig.update_layout(template=TEMPLATE, height=height + 130,
                      margin=dict(l=75, r=30, t=100, b=60),
                      legend=dict(y=1.13))
    fig.update_xaxes(tickprefix="$", title_text="spot at the horizon [$]",
                     row=2, col=1)
    fig.update_yaxes(tickprefix="$", title_text="put value", row=1, col=1)
    fig.update_yaxes(tickprefix="$", title_text="difference", row=2, col=1)
    fig.update_annotations(font_size=13)
    return fig
