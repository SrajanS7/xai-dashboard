"""
ui.py
-----
Streamlit dashboard for the XAI Heart Disease Predictor.

Panels:
  1. Dataset Explorer      — feature distributions, class balance
  2. Model Comparison      — RF vs LR metrics
  3. Global SHAP           — feature importance + beeswarm
  4. Individual Prediction — pick a patient, see explanation
  5. What-If Simulator     — tweak features, watch prediction change
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import streamlit as st

from app.data_loader import load_data, get_features_and_target, FEATURE_COLS, FEATURE_DESCRIPTIONS
from app.trainer import train_and_evaluate, load_model, load_scaler
from app.explainer import get_explainer, get_global_shap_values, explain_single
from app.visualiser import (
    plot_global_importance,
    plot_shap_beeswarm,
    plot_waterfall,
    plot_model_comparison,
    plot_feature_distribution,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="XAI Heart Disease Dashboard",
    page_icon="🫀",
    layout="wide",
)

# ── Cache expensive operations ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training models...")
def get_training_output():
    return train_and_evaluate()


@st.cache_resource(show_spinner="Loading explainer...")
def get_cached_explainer():
    return get_explainer()


@st.cache_data(show_spinner="Computing SHAP values...")
def get_cached_global_shap(_explainer, _X_test):
    return get_global_shap_values(_X_test, _explainer)


# ── Load everything ───────────────────────────────────────────────────────────

df                = load_data()
training_output   = get_training_output()
X_test            = training_output["X_test"]
y_test            = training_output["y_test"]
X_train           = training_output["X_train"]
results           = training_output["results"]
explainer         = get_cached_explainer()
global_shap       = get_cached_global_shap(explainer, X_test)
scaler            = load_scaler()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=60)
st.sidebar.title("XAI Dashboard")
st.sidebar.markdown("Heart Disease · UCI Dataset · Random Forest + SHAP")
st.sidebar.divider()

panel = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dataset Explorer",
        "🤖 Model Comparison",
        "🌍 Global SHAP",
        "🔍 Individual Prediction",
        "🎛️ What-If Simulator",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Built by Srajan Sharma · "
    "[GitHub](https://github.com/SrajanS7/xai-dashboard)"
)

# ── Panel 1: Dataset Explorer ─────────────────────────────────────────────────

if panel == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown(
        "Explore the **Heart Disease UCI dataset** — 303 patients, "
        "13 clinical features, binary target (0 = no disease, 1 = disease)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(df))
    col2.metric("Features", len(FEATURE_COLS))
    col3.metric("Disease Cases", int(df["target"].sum()))
    col4.metric("Disease Rate", f"{df['target'].mean():.1%}")

    st.divider()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Feature")
        feature = st.selectbox(
            "Select a feature to explore",
            FEATURE_COLS,
            format_func=lambda x: f"{x} — {FEATURE_DESCRIPTIONS[x][:30]}",
        )
        st.markdown(f"**{feature}**: {FEATURE_DESCRIPTIONS[feature]}")
        st.dataframe(
            df[feature].describe().round(3).to_frame(),
            use_container_width=True,
        )

    with col_right:
        fig = plot_feature_distribution(df, feature)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)


# ── Panel 2: Model Comparison ─────────────────────────────────────────────────

elif panel == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown(
        "Two classifiers trained on the same data. "
        "Logistic Regression provides an interpretable baseline; "
        "Random Forest powers the SHAP explanations."
    )

    fig = plot_model_comparison(results)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    for col, model_name in zip([col1, col2], results.keys()):
        m = results[model_name]
        with col:
            st.subheader(model_name)
            st.metric("Accuracy", m["accuracy"])
            st.metric("ROC-AUC",  m["roc_auc"])
            st.metric("F1 Score", m["f1"])
            st.metric("CV AUC",   f"{m['cv_auc_mean']} ± {m['cv_auc_std']}")
            with st.expander("Classification Report"):
                st.text(m["report"])


# ── Panel 3: Global SHAP ──────────────────────────────────────────────────────

elif panel == "🌍 Global SHAP":
    st.title("🌍 Global SHAP Explanations")
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** — how much does each feature "
        "contribute to predictions across the entire test set?"
    )

    tab1, tab2 = st.tabs(["Feature Importance", "Beeswarm Plot"])

    with tab1:
        st.markdown(
            "Mean absolute SHAP value per feature — higher = more influential overall."
        )
        fig = plot_global_importance(global_shap["mean_abs_shap"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Rankings")
        ranking = global_shap["mean_abs_shap"].reset_index()
        ranking.columns = ["Feature", "Mean |SHAP|"]
        ranking["Description"] = ranking["Feature"].map(FEATURE_DESCRIPTIONS)
        st.dataframe(ranking, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(
            "Each dot is one patient. "
            "**Colour** = feature value (red=high, blue=low). "
            "**Position** = impact on prediction."
        )
        fig = plot_shap_beeswarm(global_shap["shap_values"], global_shap["X"])
        st.plotly_chart(fig, use_container_width=True)


# ── Panel 4: Individual Prediction ───────────────────────────────────────────

elif panel == "🔍 Individual Prediction":
    st.title("🔍 Individual Prediction Explanation")
    st.markdown(
        "Select a patient from the test set and see exactly "
        "why the model made its prediction."
    )

    patient_idx = st.slider(
        "Select patient index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
    )

    row    = X_test.iloc[[patient_idx]]
    actual = int(y_test.iloc[patient_idx])
    result = explain_single(row, explainer)

    # ── Prediction summary ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    pred_label   = "🔴 Disease"    if result["prediction"] else "🟢 No Disease"
    actual_label = "🔴 Disease"    if actual               else "🟢 No Disease"
    correct      = "✅ Correct"    if result["prediction"] == actual else "❌ Wrong"

    col1.metric("Prediction",   pred_label)
    col2.metric("Actual",       actual_label)
    col3.metric("Model",        correct)

    st.metric("Probability of Disease", f"{result['probability']:.1%}")

    st.divider()

    # ── Waterfall chart ───────────────────────────────────────────────────────
    fig = plot_waterfall(
        result["shap_values"],
        result["base_value"],
        FEATURE_COLS,
        row.iloc[0],
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Top features table ────────────────────────────────────────────────────
    st.subheader("Top Contributing Features")
    top = result["top_features"].copy()
    top["direction"] = top["shap_value"].apply(
        lambda v: "↑ Increases risk" if v > 0 else "↓ Decreases risk"
    )
    top["description"] = top["feature"].map(FEATURE_DESCRIPTIONS)
    st.dataframe(
        top[["feature", "shap_value", "direction", "description"]],
        use_container_width=True,
        hide_index=True,
    )


# ── Panel 5: What-If Simulator ────────────────────────────────────────────────

elif panel == "🎛️ What-If Simulator":
    st.title("🎛️ What-If Simulator")
    st.markdown(
        "Adjust patient features using the sliders and watch the "
        "model's prediction and explanation update in real time."
    )

    # ── Feature sliders ───────────────────────────────────────────────────────
    st.subheader("Patient Features")

    # Use dataset min/max as slider bounds
    col1, col2 = st.columns(2)
    user_input = {}

    for i, feature in enumerate(FEATURE_COLS):
        col = col1 if i % 2 == 0 else col2
        with col:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            step = 1.0 if df[feature].nunique() <= 10 else 0.1

            user_input[feature] = st.slider(
                f"{feature} — {FEATURE_DESCRIPTIONS[feature][:40]}",
                min_value=min_val,
                max_value=max_val,
                value=round(mean_val, 1),
                step=step,
            )

    st.divider()

    # ── Scale input + predict ─────────────────────────────────────────────────
    input_df  = pd.DataFrame([user_input])
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=FEATURE_COLS,
    )

    result = explain_single(input_scaled, explainer)

    # ── Results ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    pred_label = "🔴 Disease" if result["prediction"] else "🟢 No Disease"
    col1.metric("Prediction",              pred_label)
    col2.metric("Probability of Disease",  f"{result['probability']:.1%}")

    # Colour the probability bar
    prob = result["probability"]
    bar_colour = "#EF553B" if prob > 0.5 else "#636EFA"
    st.markdown(
        f"""
        <div style="background:{bar_colour};width:{prob*100:.1f}%;
        height:20px;border-radius:10px;margin-bottom:16px;"></div>
        """,
        unsafe_allow_html=True,
    )

    # ── Waterfall ─────────────────────────────────────────────────────────────
    fig = plot_waterfall(
        result["shap_values"],
        result["base_value"],
        FEATURE_COLS,
        input_df.iloc[0],
    )
    st.plotly_chart(fig, use_container_width=True)