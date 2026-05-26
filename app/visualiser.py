"""
visualiser.py
-------------
Builds all Plotly charts used in the XAI dashboard:

  1. plot_global_importance  → horizontal bar chart of mean |SHAP| values
  2. plot_shap_beeswarm      → beeswarm-style dot plot (SHAP vs feature value)
  3. plot_waterfall          → individual prediction waterfall chart
  4. plot_model_comparison   → grouped bar chart of Accuracy / AUC / F1
  5. plot_feature_distribution → histogram for dataset explorer

All functions return a plotly.graph_objects.Figure ready for st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from app.data_loader import FEATURE_DESCRIPTIONS

# ── Colour palette ────────────────────────────────────────────────────────────

POSITIVE_COLOUR = "#EF553B"   # red   — pushes toward disease
NEGATIVE_COLOUR = "#636EFA"   # blue  — pushes away from disease
NEUTRAL_COLOUR  = "#00CC96"   # green — used for bar charts / comparisons


# ── 1. Global feature importance ──────────────────────────────────────────────

def plot_global_importance(mean_abs_shap: pd.Series) -> go.Figure:
    """
    Horizontal bar chart of mean absolute SHAP values.
    Most important feature at the top.

    Parameters
    ----------
    mean_abs_shap : pd.Series — index=feature names, values=mean |SHAP|
    """
    df = mean_abs_shap.reset_index()
    df.columns = ["feature", "importance"]
    df = df.sort_values("importance")  # ascending so top feature is at top

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color=NEUTRAL_COLOUR,
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── 2. Beeswarm plot ──────────────────────────────────────────────────────────

def plot_shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame) -> go.Figure:
    """
    Beeswarm-style scatter plot: SHAP value (x) vs feature (y),
    coloured by the actual feature value (low=blue, high=red).

    Parameters
    ----------
    shap_values : np.ndarray — shape (n_samples, n_features)
    X           : pd.DataFrame — original scaled feature matrix
    """
    feature_names = X.columns.tolist()
    # Sort features by mean |SHAP| so most important is at top
    order = np.argsort(np.abs(shap_values).mean(axis=0))

    fig = go.Figure()

    for i in order:
        fname  = feature_names[i]
        sv     = shap_values[:, i]
        fv     = X.iloc[:, i].values

        # Normalise feature values to [0,1] for colour mapping
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)

        # Add jitter on y-axis to avoid overplotting
        jitter = np.random.uniform(-0.3, 0.3, size=len(sv))

        fig.add_trace(go.Scatter(
            x=sv,
            y=np.full_like(sv, i, dtype=float) + jitter,
            mode="markers",
            marker=dict(
                size=4,
                color=fv_norm,
                colorscale="RdBu_r",
                opacity=0.6,
            ),
            name=fname,
            hovertemplate=(
                f"<b>{fname}</b><br>"
                "SHAP: %{x:.4f}<br>"
                "Feature value: %{text}<extra></extra>"
            ),
            text=[f"{v:.2f}" for v in fv],
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1)

    fig.update_layout(
        title="SHAP Beeswarm Plot",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickvals=list(range(len(feature_names))),
            ticktext=feature_names,
        ),
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── 3. Waterfall chart ────────────────────────────────────────────────────────

def plot_waterfall(shap_flat: np.ndarray,
                   base_value: float,
                   feature_names: list,
                   feature_values: pd.Series) -> go.Figure:
    """
    Waterfall chart for a single prediction.
    Shows how each feature pushes the prediction up or down from baseline.

    Parameters
    ----------
    shap_flat      : np.ndarray — shape (13,) SHAP values for one sample
    base_value     : float — model expected value (baseline)
    feature_names  : list of feature name strings
    feature_values : pd.Series — actual feature values for this patient
    """
    # Sort by absolute SHAP, show top 8 for readability
    indices = np.argsort(np.abs(shap_flat))[::-1][:8]
    sv      = shap_flat[indices]
    names   = [f"{feature_names[i]}={feature_values.iloc[i]:.2f}"
               for i in indices]
    colours = [POSITIVE_COLOUR if v > 0 else NEGATIVE_COLOUR for v in sv]

    # Build cumulative values for waterfall
    cumulative = [base_value]
    for v in sv:
        cumulative.append(cumulative[-1] + v)
    final = cumulative[-1]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(sv) + ["total"],
        x=["Base"] + names + ["Prediction"],
        y=[base_value] + list(sv) + [final],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": NEGATIVE_COLOUR}},
        increasing={"marker": {"color": POSITIVE_COLOUR}},
        totals={"marker": {"color": "#FFA15A"}},
        hovertemplate="%{x}<br>SHAP: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Individual Prediction — Waterfall Explanation",
        yaxis_title="Model Output (probability)",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── 4. Model comparison ───────────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> go.Figure:
    """
    Grouped bar chart comparing Accuracy, AUC, and F1
    for Random Forest vs Logistic Regression.

    Parameters
    ----------
    results : dict — output from trainer.train_and_evaluate()['results']
    """
    metrics = ["accuracy", "roc_auc", "f1"]
    labels  = ["Accuracy", "ROC-AUC", "F1"]
    colours = [NEUTRAL_COLOUR, POSITIVE_COLOUR, NEGATIVE_COLOUR]
    models  = list(results.keys())

    fig = go.Figure()

    for metric, label, colour in zip(metrics, labels, colours):
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=[results[m][metric] for m in models],
            marker_color=colour,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Model Comparison — Accuracy / ROC-AUC / F1",
        yaxis=dict(title="Score", range=[0, 1]),
        barmode="group",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── 5. Feature distribution ───────────────────────────────────────────────────

def plot_feature_distribution(df: pd.DataFrame, feature: str) -> go.Figure:
    """
    Histogram of a feature split by target class.
    Used in the Dataset Explorer panel.

    Parameters
    ----------
    df      : pd.DataFrame — full dataset including 'target' column
    feature : str — column name to plot
    """
    description = FEATURE_DESCRIPTIONS.get(feature, feature)

    fig = go.Figure()

    for target, label, colour in [(0, "No Disease", NEGATIVE_COLOUR),
                                   (1, "Disease",    POSITIVE_COLOUR)]:
        subset = df[df["target"] == target][feature]
        fig.add_trace(go.Histogram(
            x=subset,
            name=label,
            marker_color=colour,
            opacity=0.7,
            hovertemplate=f"{label}<br>{feature}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{feature} — {description}",
        xaxis_title=feature,
        yaxis_title="Count",
        barmode="overlay",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig