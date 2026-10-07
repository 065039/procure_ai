import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from utils.data_loader import load_data_source
from utils.validation import validate_dataset
from utils.scoring import evaluate_vendors, compare_rankings
from utils.ai_engine import generate_ai_explanation

load_dotenv()

st.set_page_config(
    page_title="ProcureAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = [
    "vendor_id", "vendor_name", "category", "price_lakhs",
    "quality_score", "lead_time_days", "on_time_delivery_pct",
    "reliability_score", "defect_rate_pct", "sustainability_score",
    "years_in_business", "esg_certified"
]

DEFAULT_WEIGHTS = {
    "Cost": 30.0,
    "Quality": 30.0,
    "Delivery": 20.0,
    "Reliability": 15.0,
    "Sustainability": 5.0,
}

if "weights" not in st.session_state:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()
if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = None
if "scenario_results" not in st.session_state:
    st.session_state.scenario_results = None
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "source_name" not in st.session_state:
    st.session_state.source_name = "Not loaded"


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def load_and_validate(df, source_name):
    validation = validate_dataset(df, REQUIRED_COLUMNS)
    if not validation["valid"]:
        st.error("Dataset validation failed.")
        for msg in validation["errors"]:
            st.error(msg)
        return False

    if validation["warnings"]:
        for msg in validation["warnings"]:
            st.warning(msg)

    st.session_state.data = df.copy()
    st.session_state.source_name = source_name
    st.session_state.results = None
    st.session_state.scenario_results = None
    st.session_state.ai_result = None
    return True


with st.sidebar:
    st.markdown("# 📊 ProcureAI")
    st.caption("AI-Powered Vendor Selection & Procurement Decision Support")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Vendor Data", "Decision Criteria", "Evaluation",
         "AI Insight", "What-If Analysis", "Methodology"],
    )

    st.divider()
    st.caption("Architecture")
    st.caption("Ranking: Deterministic Python")
    st.caption("Explanation: OpenRouter Free LLM")
    st.caption("Data: Synthetic / user-uploaded")

    if st.button("Reset Application", use_container_width=True):
        for key in ["data", "results", "scenario_results", "ai_result"]:
            st.session_state[key] = None
        st.session_state.weights = DEFAULT_WEIGHTS.copy()
        st.session_state.source_name = "Not loaded"
        st.rerun()


if page == "Overview":
    st.title("ProcureAI")
    st.subheader("AI-Powered Vendor Selection & Procurement Decision-Support Application")
    st.write(
        "ProcureAI helps procurement teams compare suppliers across cost, quality, "
        "delivery, reliability and sustainability. It combines transparent weighted "
        "analytics with generative AI explanations."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Decision Criteria", "5")
    with c2:
        metric_card("Default Weights", "100%")
    with c3:
        metric_card("Ranking Engine", "Python")
    with c4:
        metric_card("AI Layer", "OpenRouter")

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. **Load vendor data** using the built-in synthetic dataset or CSV/XLSX upload.
    2. **Validate the data** for required fields and sensible ranges.
    3. **Set decision weights** according to procurement priorities.
    4. **Calculate normalized scores and rankings** using a deterministic model.
    5. **Inspect score contributions and trade-offs** through visual analytics.
    6. **Generate an AI explanation** using OpenRouter without allowing the LLM to change the ranking.
    7. **Run what-if scenarios** to understand how recommendation sensitivity changes.
    """)

    st.info(
        "Human-in-the-loop principle: ProcureAI recommends based on the supplied data and "
        "weights; the procurement professional remains responsible for the final decision."
    )

    if st.session_state.data is None:
        st.markdown("### Get started")
        if st.button("Load Sample Vendor Dataset", type="primary"):
            try:
                df = load_data_source(None, use_sample=True)
                if load_and_validate(df, "Built-in sample dataset"):
                    st.success("Sample dataset loaded.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not load sample dataset: {exc}")
    else:
        st.success(f"Dataset loaded: {st.session_state.source_name}")


elif page == "Vendor Data":
    st.title("Vendor Data")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Use Sample Dataset", type="primary", use_container_width=True):
            try:
                df = load_data_source(None, use_sample=True)
                if load_and_validate(df, "Built-in sample dataset"):
                    st.success("Sample dataset loaded.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not load sample dataset: {exc}")

    with col2:
        uploaded = st.file_uploader(
            "Upload vendor data",
            type=["csv", "xlsx"],
            help="Use a CSV or Excel file containing the required vendor fields.",
        )
        if uploaded is not None:
            try:
                df = load_data_source(uploaded, use_sample=False)
                if load_and_validate(df, uploaded.name):
                    st.success(f"Loaded {len(df)} vendors from {uploaded.name}.")
            except Exception as exc:
                st.error(f"Could not read the uploaded file: {exc}")

    if st.session_state.data is None:
        st.info("Load the sample dataset or upload a CSV/XLSX file to continue.")
    else:
        df = st.session_state.data
        st.markdown(f"**Source:** {st.session_state.source_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vendors", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Cells", int(df.isna().sum().sum()))
        c4.metric("Categories", df["category"].nunique())

        st.markdown("### Filters")
        f1, f2 = st.columns(2)
        with f1:
            categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
            selected_category = st.selectbox("Category", categories)
        with f2:
            esg_options = ["All"] + sorted(df["esg_certified"].astype(str).unique().tolist())
            selected_esg = st.selectbox("ESG Certified", esg_options)

        filtered = df.copy()
        if selected_category != "All":
            filtered = filtered[filtered["category"] == selected_category]
        if selected_esg != "All":
            filtered = filtered[filtered["esg_certified"].astype(str) == selected_esg]

        st.markdown(f"### Data Preview ({len(filtered)} vendors)")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Current Data",
            data=csv_bytes,
            file_name="procureai_vendor_data.csv",
            mime="text/csv",
        )


elif page == "Decision Criteria":
    st.title("Decision Criteria")
    st.write("Adjust the relative importance of each criterion. The total must equal 100%.")

    labels = list(DEFAULT_WEIGHTS.keys())
    cols = st.columns(5)
    new_weights = {}

    for i, label in enumerate(labels):
        with cols[i]:
            new_weights[label] = st.slider(
                label,
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.weights[label]),
                step=5.0,
            )

    total = sum(new_weights.values())
    st.metric("Total Weight", f"{total:.0f}%")

    if abs(total - 100.0) < 0.01:
        st.success("Weights are valid.")
        st.session_state.weights = new_weights
    else:
        st.error("Weights must add up to exactly 100%.")

    if st.button("Reset to Default Weights"):
        st.session_state.weights = DEFAULT_WEIGHTS.copy()
        st.rerun()

    st.markdown("### Current weighting logic")
    st.dataframe(
        pd.DataFrame(
            {
                "Criterion": list(st.session_state.weights.keys()),
                "Weight (%)": list(st.session_state.weights.values()),
                "Direction": [
                    "Lower is better",
                    "Higher is better",
                    "Lower lead time + higher on-time delivery",
                    "Higher is better",
                    "Higher is better",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.session_state.data is None:
        st.info("Load vendor data before running an evaluation.")


elif page == "Evaluation":
    st.title("Vendor Evaluation")

    if st.session_state.data is None:
        st.warning("Load vendor data first from the Vendor Data page.")
    else:
        weights = st.session_state.weights
        if abs(sum(weights.values()) - 100) >= 0.01:
            st.error("Invalid weights. Go to Decision Criteria and make the total 100%.")
        else:
            if st.button("Run Vendor Evaluation", type="primary"):
                try:
                    st.session_state.results = evaluate_vendors(
                        st.session_state.data, weights
                    )
                    st.session_state.ai_result = None
                    st.success("Evaluation completed.")
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

            if st.session_state.results is not None:
                results = st.session_state.results
                top = results.iloc[0]

                st.markdown("## 🏆 Recommended Vendor")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Vendor", top["vendor_name"])
                c2.metric("Overall Score", f'{top["overall_score"]:.1f}/100')
                c3.metric("Rank", f'#{int(top["rank"])}')
                c4.metric("Category", top["category"])

                st.markdown("### Top Vendors")
                display_cols = [
                    "rank", "vendor_name", "category", "overall_score",
                    "cost_score", "quality_score", "delivery_score",
                    "reliability_score", "sustainability_score"
                ]
                st.dataframe(
                    results[display_cols].head(10).style.format({
                        "overall_score": "{:.1f}",
                        "cost_score": "{:.1f}",
                        "quality_score": "{:.1f}",
                        "delivery_score": "{:.1f}",
                        "reliability_score": "{:.1f}",
                        "sustainability_score": "{:.1f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                chart = px.bar(
                    results.head(10).sort_values("overall_score"),
                    x="overall_score",
                    y="vendor_name",
                    orientation="h",
                    title="Top 10 Vendor Ranking",
                    labels={"overall_score": "Overall Score", "vendor_name": "Vendor"},
                )
                chart.update_xaxes(range=[0, 100])
                st.plotly_chart(chart, use_container_width=True)

                st.markdown("### Score Contribution — Recommended Vendor")
                contribution_cols = [
                    "cost_contribution", "quality_contribution",
                    "delivery_contribution", "reliability_contribution",
                    "sustainability_contribution"
                ]
                contribution_labels = [
                    "Cost", "Quality", "Delivery", "Reliability", "Sustainability"
                ]
                contribution_df = pd.DataFrame({
                    "Criterion": contribution_labels,
                    "Contribution": [float(top[c]) for c in contribution_cols]
                })
                contribution_chart = px.bar(
                    contribution_df,
                    x="Criterion",
                    y="Contribution",
                    title=f"Weighted Contribution — {top['vendor_name']}",
                    labels={"Contribution": "Points contributed"},
                )
                st.plotly_chart(contribution_chart, use_container_width=True)

                st.markdown("### Top 3 Criteria Comparison")
                comparison = results.head(3).copy()
                score_cols = ["cost_score", "quality_score", "delivery_score",
                              "reliability_score", "sustainability_score"]
                melted = comparison.melt(
                    id_vars=["vendor_name"],
                    value_vars=score_cols,
                    var_name="Criterion",
                    value_name="Score"
                )
                melted["Criterion"] = melted["Criterion"].str.replace("_score", "", regex=False).str.title()
                comparison_chart = px.bar(
                    melted,
                    x="Criterion",
                    y="Score",
                    color="vendor_name",
                    barmode="group",
                    title="Top 3 Vendors — Criterion Scores",
                    range_y=[0, 100],
                )
                st.plotly_chart(comparison_chart, use_container_width=True)

                st.markdown("### Vendor Detail")
                selected_vendor = st.selectbox(
                    "Select a vendor",
                    results["vendor_name"].tolist()
                )
                selected_row = results[results["vendor_name"] == selected_vendor].iloc[0]
                st.json({
                    "vendor": selected_vendor,
                    "rank": int(selected_row["rank"]),
                    "overall_score": round(float(selected_row["overall_score"]), 2),
                    "criterion_scores": {
                        "cost": round(float(selected_row["cost_score"]), 2),
                        "quality": round(float(selected_row["quality_score"]), 2),
                        "delivery": round(float(selected_row["delivery_score"]), 2),
                        "reliability": round(float(selected_row["reliability_score"]), 2),
                        "sustainability": round(float(selected_row["sustainability_score"]), 2),
                    }
                })

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Evaluation Results",
                    data=csv,
                    file_name="procureai_evaluation_results.csv",
                    mime="text/csv",
                )


elif page == "AI Insight":
    st.title("AI Procurement Insight")

    if st.session_state.results is None:
        st.info("Run a vendor evaluation first.")
    else:
        results = st.session_state.results
        weights = st.session_state.weights
        top = results.iloc[0]
        runner_up = results.iloc[1] if len(results) > 1 else None

        if st.button("Generate AI Explanation", type="primary"):
            with st.spinner("Generating procurement insight..."):
                st.session_state.ai_result = generate_ai_explanation(
                    top_vendor=top.to_dict(),
                    runner_up=runner_up.to_dict() if runner_up is not None else None,
                    weights=weights,
                )

        if st.session_state.ai_result:
            result = st.session_state.ai_result
            if result["success"]:
                st.markdown("### 🤖 AI Procurement Insight")
                st.markdown(result["text"])
                st.caption("OpenRouter is used for explanation only; the ranking is calculated by the Python scoring engine.")
            else:
                st.warning(result["message"])
                st.info("The deterministic vendor ranking remains available on the Evaluation page.")
        else:
            st.info("Click 'Generate AI Explanation' to analyze the current recommendation.")

        st.markdown("### Decision-support disclaimer")
        st.caption(
            "AI-generated text is an interpretation of calculated results. Verify supplier "
            "information and business constraints before making a final procurement decision."
        )


elif page == "What-If Analysis":
    st.title("What-If Analysis")
    st.write("Test how changes in procurement priorities affect the recommendation.")

    if st.session_state.data is None:
        st.warning("Load vendor data first.")
    elif st.session_state.results is None:
        st.info("Run the base evaluation first.")
    else:
        base_results = st.session_state.results
        base_top = base_results.iloc[0]

        st.markdown(f"**Current recommendation:** {base_top['vendor_name']} "
                    f"({base_top['overall_score']:.1f}/100)")

        st.markdown("### Scenario Weights")
        cols = st.columns(5)
        scenario_weights = {}
        for i, label in enumerate(DEFAULT_WEIGHTS.keys()):
            with cols[i]:
                scenario_weights[label] = st.number_input(
                    label,
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.weights[label]),
                    step=5.0,
                    key=f"scenario_{label}",
                )

        total = sum(scenario_weights.values())
        st.metric("Scenario Total", f"{total:.0f}%")

        if abs(total - 100) >= 0.01:
            st.error("Scenario weights must add up to 100%.")
        elif st.button("Run Scenario", type="primary"):
            try:
                scenario = evaluate_vendors(st.session_state.data, scenario_weights)
                st.session_state.scenario_results = scenario
            except Exception as exc:
                st.error(f"Scenario evaluation failed: {exc}")

        if st.session_state.scenario_results is not None:
            scenario = st.session_state.scenario_results
            scenario_top = scenario.iloc[0]

            changed = base_top["vendor_name"] != scenario_top["vendor_name"]
            if changed:
                st.warning(
                    f"Recommendation changed: {base_top['vendor_name']} → {scenario_top['vendor_name']}"
                )
                st.write(
                    "The recommendation changed because the relative importance of the decision "
                    "criteria changed."
                )
            else:
                st.success(
                    f"Recommendation remains {scenario_top['vendor_name']} under the scenario."
                )

            comparison = compare_rankings(base_results, scenario)
            st.dataframe(comparison.head(10), use_container_width=True, hide_index=True)

            chart_data = pd.DataFrame({
                "Vendor": base_results.head(5)["vendor_name"].tolist(),
                "Base Score": base_results.head(5)["overall_score"].tolist(),
            })
            scenario_lookup = scenario.set_index("vendor_name")["overall_score"]
            chart_data["Scenario Score"] = chart_data["Vendor"].map(scenario_lookup)

            melted = chart_data.melt(
                id_vars="Vendor",
                var_name="Scenario",
                value_name="Score"
            )
            chart = px.bar(
                melted,
                x="Vendor",
                y="Score",
                color="Scenario",
                barmode="group",
                title="Base vs Scenario — Top 5 Base Vendors",
                range_y=[0, 100],
            )
            st.plotly_chart(chart, use_container_width=True)


elif page == "Methodology":
    st.title("Methodology & Responsible AI")

    st.markdown("## Architecture")
    st.code("""
Vendor Data
    ↓
Validation
    ↓
Normalization
    ↓
Weighted Multi-Criteria Scoring
    ↓
Vendor Ranking
    ↓
Score Contributions + Visual Analytics
    ↓
OpenRouter Explanation Layer
    ↓
Human Procurement Decision
""")

    st.markdown("## Scoring methodology")
    st.markdown("""
    **Cost:** lower price is better.

    **Quality:** higher quality score is better.

    **Delivery:** combines on-time delivery and inverse lead time.

    **Reliability:** higher reliability score is better.

    **Sustainability:** higher sustainability score is better.

    Each criterion is normalized to a 0–100 scale and multiplied by the user-selected weight.
    The weighted sum produces the final vendor score.
    """)

    st.markdown("## Why OpenRouter is not the ranking engine")
    st.write(
        "The ranking is deliberately calculated outside the LLM so that the recommendation is "
        "reproducible, auditable and directly tied to the selected weights. OpenRouter is used only "
        "to explain the calculated result in business language."
    )

    st.markdown("## Guardrails")
    st.markdown("""
    - The LLM receives calculated results rather than raw instructions to choose a vendor.
    - The prompt prohibits invented vendor facts.
    - The LLM cannot modify the Python ranking.
    - Missing information should be acknowledged rather than fabricated.
    - API failure does not remove the deterministic ranking.
    - Final procurement decisions remain with a human.
    """)

    st.markdown("## Limitations")
    st.markdown("""
    1. Results depend on data quality.
    2. The chosen criteria and weights may not capture every real-world procurement factor.
    3. The prototype uses synthetic vendor data.
    4. Contract terms, supplier relationships, geopolitical risks and financial stability are not fully modeled.
    5. Generative AI can produce imperfect explanations and therefore requires human review.
    """)

    st.markdown("## Responsible-use statement")
    st.info(
        "ProcureAI is a decision-support prototype. It should not autonomously select, contract, "
        "or onboard a supplier. Users should independently verify supplier information and relevant "
        "commercial, legal, operational and compliance considerations."
    )
