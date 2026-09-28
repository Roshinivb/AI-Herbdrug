
from pathlib import Path
import sys
import contextlib
import io

import streamlit as st
import pandas as pd

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the existing trained model and prediction functions.
# Step 1 is required so main.py does not start its input loop.
from main import (
    gnn_predict,
    get_ddid_record,
    herb_to_id,
    drug_to_id,
    ddid,
)

from model.biological_features import (
    get_herb_features,
    get_drug_features,
    get_pair_evidence,
)

from model.unseen_predictor import predict_unseen


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Herb-Drug Interaction Predictor",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.15rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🌿 AI Herb-Drug Interaction Prediction System'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Graph Neural Network predictions, DDID records, '
    'biological evidence and optional molecular docking.'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Research prototype only. Predictions are not medical advice "
    "and should not be used to start, stop, or change medication."
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.header("About the project")

    st.write(
        "This system analyses a selected herb-drug pair using "
        "a trained GCN model and available evidence."
    )

    st.markdown("**Analysis modules**")
    st.write("• GCN risk classification")
    st.write("• DDID dataset lookup")
    st.write("• Biological evidence")
    st.write("• Optional AutoDock Vina analysis")

    st.divider()

    st.caption(
        "The model's displayed confidence is its predicted "
        "class probability, not a clinically calibrated risk."
    )


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

st.subheader("1. Select an herb and a drug")

left, right = st.columns(2)

herb_options = sorted(herb_to_id.keys(), key=str.lower)
drug_options = sorted(drug_to_id.keys(), key=str.lower)

with left:
    herb = st.selectbox(
        "Select herb",
        options=herb_options,
        index=None,
        placeholder="Search for an herb...",
    )

with right:
    drug = st.selectbox(
        "Select pharmaceutical drug",
        options=drug_options,
        index=None,
        placeholder="Search for a drug...",
    )

run_docking = st.checkbox(
    "Also run biological analysis and molecular docking",
    value=False,
    help=(
        "This can take additional time. Docking runs only when "
        "the selected evidence identifies a supported ligand "
        "and CYP3A4/CYP3A5 receptor."
    ),
)

predict_clicked = st.button(
    "🔬 Analyse interaction",
    type="primary",
    use_container_width=True,
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def display_list(items):
    """Convert a list or set to readable text."""
    if not items:
        return "No data available"
    return ", ".join(str(item) for item in sorted(items, key=str))


def show_risk(risk):
    """Display a risk label without implying clinical certainty."""
    risk = str(risk).upper()

    if risk == "HIGH":
        st.error("Predicted risk category: HIGH")
    elif risk == "MEDIUM":
        st.warning("Predicted risk category: MEDIUM")
    elif risk == "LOW":
        st.success("Predicted risk category: LOW")
    else:
        st.info(f"Risk category: {risk}")


def show_probability_chart(prediction):
    """Display the three GNN class probabilities."""
    probability_data = pd.DataFrame(
        {
            "Risk category": ["LOW", "MEDIUM", "HIGH"],
            "Model probability (%)": [
                prediction["low"],
                prediction["medium"],
                prediction["high"],
            ],
        }
    ).set_index("Risk category")

    st.bar_chart(probability_data)

    st.dataframe(
        probability_data,
        use_container_width=True,
    )


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

if predict_clicked:

    if not herb or not drug:
        st.warning("Please select both an herb and a drug.")

    else:
        st.divider()

        st.subheader("2. GNN prediction")

        with st.spinner("Running the trained GNN model..."):
            try:
                prediction = gnn_predict(herb, drug)
            except Exception as exc:
                prediction = None
                st.error(f"GNN prediction failed: {exc}")

        if prediction is None:
            st.warning(
                "The selected herb-drug pair could not be mapped "
                "to the trained model. Please check the dataset "
                "and the model's entity-ID mapping."
            )

        else:
            metric1, metric2 = st.columns(2)

            with metric1:
                show_risk(prediction["risk"])

            with metric2:
                st.metric(
                    "Predicted-class probability",
                    f'{prediction["confidence"]:.2f}%',
                )

            st.caption(
                "This percentage is the model's softmax probability "
                "for its predicted class; it is not the probability "
                "that a clinical interaction will occur."
            )

            st.markdown("#### Class probabilities")
            show_probability_chart(prediction)

        # ----------------------------------------------------
        # DDID RECORD
        # ----------------------------------------------------

        st.divider()
        st.subheader("3. DDID dataset evidence")

        try:
            record = get_ddid_record(herb, drug)
        except Exception as exc:
            record = None
            st.error(f"Could not retrieve the DDID record: {exc}")

        if record is not None:
            record_dict = (
                record.to_dict()
                if hasattr(record, "to_dict")
                else dict(record)
            )

            c1, c2 = st.columns(2)

            with c1:
                st.write("**Recorded risk**")
                st.write(str(record_dict.get("Risk", "Not available")))

                st.write("**Recorded effects**")
                st.write(str(record_dict.get("Effects", "Not available")))

            with c2:
                st.write("**Evidence count**")
                st.write(
                    str(record_dict.get("Evidence_Count", "Not available"))
                )

                st.write("**PubMed IDs (PMIDs)**")
                st.write(str(record_dict.get("PMIDs", "Not available")))

            st.caption(
                "This is the recorded dataset evidence for the exact "
                "pair, when available. It is separate from the GNN output."
            )

        else:
            st.info(
                "No exact DDID record was found for this pair. "
                "This does not establish that the combination is safe."
            )

        # ----------------------------------------------------
        # BIOLOGICAL FEATURES
        # ----------------------------------------------------

        st.divider()
        st.subheader("4. Biological features")

        try:
            herb_features = get_herb_features(herb)
            drug_features = get_drug_features(drug)
            pair_evidence = get_pair_evidence(herb, drug)

            common_targets = sorted(
                set(herb_features.get("proteins", []))
                & set(drug_features.get("targets", [])),
                key=str,
            )

            feature_left, feature_right = st.columns(2)

            with feature_left:
                st.markdown("#### Herb")
                st.write(f"**Name:** {herb}")
                st.write(
                    "**Compounds:** "
                    + display_list(herb_features.get("compounds", []))
                )
                st.write(
                    "**Targets/proteins:** "
                    + display_list(herb_features.get("proteins", []))
                )

            with feature_right:
                st.markdown("#### Drug")
                st.write(f"**Name:** {drug}")
                st.write(
                    "**Targets:** "
                    + display_list(drug_features.get("targets", []))
                )

                st.write(
                    "**Common targets:** "
                    + display_list(common_targets)
                )

            st.markdown("#### Pair-specific biological evidence")

            if pair_evidence:
                evidence_df = pd.DataFrame(pair_evidence)
                st.dataframe(
                    evidence_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No exact pair-specific biological evidence was "
                    "found in the loaded external evidence dataset."
                )

        except Exception as exc:
            st.error(f"Could not load biological features: {exc}")

        # ----------------------------------------------------
        # OPTIONAL BIOLOGICAL PREDICTION + DOCKING
        # ----------------------------------------------------

        if run_docking:
            st.divider()
            st.subheader("5. Biological analysis and docking")

            st.warning(
                "This is a separate evidence-based analysis. "
                "Its result may differ from the GNN prediction. "
                "Docking affinity is a computational estimate, "
                "not proof of a clinical interaction."
            )

            output_buffer = io.StringIO()

            try:
                with st.spinner(
                    "Checking evidence and running supported docking..."
                ):
                    with contextlib.redirect_stdout(output_buffer):
                        biological_result = predict_unseen(herb, drug)

                output_text = output_buffer.getvalue()

                if biological_result:
                    st.markdown("#### Biological prediction")

                    biological_risk = biological_result.get("risk")
                    if biological_risk:
                        show_risk(biological_risk)

                    st.write(
                        "**Reason:** "
                        + str(biological_result.get("reason", "Not available"))
                    )

                    docking_result = biological_result.get("docking")

                    st.markdown("#### Molecular docking")

                    if docking_result:
                        affinity = docking_result.get("affinity")
                        target = docking_result.get("target")
                        ligand = docking_result.get("ligand")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Affinity (kcal/mol)",
                                (
                                    f"{affinity:.3f}"
                                    if isinstance(affinity, (float, int))
                                    else "Not available"
                                ),
                            )

                        with col2:
                            st.write("**Target**")
                            st.write(str(target or "Not available"))

                        with col3:
                            st.write("**Ligand file**")
                            st.write(
                                Path(ligand).name
                                if ligand
                                else "Not available"
                            )

                        output_path = docking_result.get("output")
                        if output_path:
                            st.caption(
                                f"Docking output: {output_path}"
                            )

                    else:
                        st.info(
                            "No docking result was produced. The required "
                            "ligand, receptor, supported target, or Vina "
                            "execution may be unavailable for this pair."
                        )

                else:
                    st.info(
                        "The biological analysis did not return a result."
                    )

                with st.expander("View docking console output"):
                    st.text(
                        output_text
                        if output_text.strip()
                        else "No console output was produced."
                    )

            except Exception as exc:
                st.error(f"Biological/docking analysis failed: {exc}")

                with st.expander("View captured console output"):
                    st.text(output_buffer.getvalue())

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.divider()
        st.caption(
            "Research limitation: the available datasets and model "
            "do not establish clinical safety. A low predicted category "
            "does not rule out an interaction. Consult a qualified "
            "healthcare professional for medication decisions."
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "AI-Enabled Herb-Drug Interaction Prediction System | "
    "GCN + DDID evidence + biological evidence + optional docking"
)