"""Streamlit front end for the predictive maintenance copilot.

Two modes (sidebar radio):
  * Equipment Monitor  — pick a machine, see its live anomaly/severity/sensors.
  * Diagnostic Copilot — paste a log entry, the agent diagnoses and grounds it.

The heavy lifting lives in the other modules; this file is presentation only:
  anomaly_detector.check_anomaly()  -> Monitor mode
  agent.run_agent()                 -> Copilot mode
"""

import os
import re

import pandas as pd
import streamlit as st

# These imports pull in the model/agent stack. anomaly_detector loads the CSV
# at import; agent builds the Bedrock client + LangGraph graph at import.
from app.anomaly_detector import check_anomaly, get_recent_trend
from app.agent import run_agent

# --------------------------------------------------------------------------- #
# Page config + light professional styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Predictive Maintenance Copilot",
    page_icon="🛠️",
    layout="wide",
)

# A little CSS for a clean, professional look (cards + severity badges).
st.markdown("""
<style>
  .pmc-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    background: rgba(128,128,128,0.04);
  }
  .pmc-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    color: white;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-size: 0.95rem;
  }
  .pmc-section-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8a8a8a;
    margin-bottom: 0.2rem;
  }
</style>
""", unsafe_allow_html=True)

# Map each severity to (colour, emoji, label). Drives the colour indicator.
SEVERITY_STYLE = {
    "normal":   ("#1a7f37", "🟢", "NORMAL"),
    "warning":  ("#bf8700", "🟡", "WARNING"),
    "critical": ("#cf222e", "🔴", "CRITICAL"),
}

PROCESSED_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "processed", "cmapss_processed.csv",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_equipment_ids(n: int = 10) -> list:
    """Pre-populate the dropdown with the first N engine IDs from CMAPSS."""
    try:
        df = pd.read_csv(PROCESSED_CSV, usecols=["engine_id"])
        ids = sorted(df["engine_id"].unique())[:n]
        return [str(i) for i in ids]
    except Exception:
        # If the processed data isn't built yet, fall back to a sensible range.
        return [str(i) for i in range(1, n + 1)]


def severity_badge(severity: str) -> str:
    """Return an HTML badge coloured by severity."""
    colour, emoji, label = SEVERITY_STYLE.get(
        severity, ("#6e7781", "⚪", str(severity).upper())
    )
    return (f'<span class="pmc-badge" style="background:{colour}">'
            f'{emoji} {label}</span>')


def split_hypothesis_action(text: str):
    """Best-effort split of the agent's diagnosis into hypothesis vs action.

    run_agent() returns one free-text diagnosis that contains both the root
    cause and the recommended fix. We split on the first 'recommend/resolution/
    action' marker so the UI can label the two sections the brief asks for.
    If no marker is found, everything is treated as the hypothesis.
    """
    if not text:
        return "", ""
    marker = re.search(
        r"(recommended action|recommendation|recommended|resolution|"
        r"suggested action|action[:\-])",
        text, re.IGNORECASE,
    )
    if marker:
        cut = marker.start()
        return text[:cut].strip(), text[cut:].strip()
    return text.strip(), ""


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🛠️ Maintenance Copilot")
mode = st.sidebar.radio(
    "Mode",
    ["Equipment Monitor", "Diagnostic Copilot"],
    help="Monitor watches a machine's health; Copilot diagnoses a fault.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("About This Tool")
st.sidebar.markdown(
    "This **Predictive Maintenance Copilot** helps engineers catch and diagnose "
    "equipment faults early. It reads live sensor health, classifies fault "
    "types from maintenance notes, retrieves similar past incidents, and an AI "
    "agent proposes a grounded root cause — **self-checking its own confidence** "
    "before answering.\n\n"
    "⚠️ **Disclaimer:** All MES / sensor / maintenance data here is **simulated** "
    "(NASA CMAPSS + synthetic logs) for a **portfolio demonstration**. Do not use "
    "for real maintenance decisions."
)


# =========================================================================== #
# MODE 1 — Equipment Monitor
# =========================================================================== #
if mode == "Equipment Monitor":
    st.title("Equipment Monitor")
    st.caption("Live anomaly signal and sensor snapshot for a selected machine.")

    equipment_ids = load_equipment_ids()
    eid = st.selectbox("Select equipment ID", equipment_ids)

    if eid:
        result = check_anomaly(eid)

        # Unknown equipment guard.
        if "note" in result and "current_rul_estimate" not in result:
            st.warning(result["note"])
        else:
            severity = result.get("anomaly_severity", "normal")
            colour = SEVERITY_STYLE.get(severity, ("#6e7781",))[0]

            # --- top row: status badge + headline metrics ---
            c1, c2, c3 = st.columns([1.4, 1, 1])
            with c1:
                st.markdown('<div class="pmc-section-title">Status</div>',
                            unsafe_allow_html=True)
                st.markdown(severity_badge(severity), unsafe_allow_html=True)
            with c2:
                st.metric("Remaining Useful Life",
                          f"{result['current_rul_estimate']} cyc")
            with c3:
                st.metric("Anomaly Detected",
                          "Yes" if result["anomaly_detected"] else "No")

            # --- coloured banner reinforcing the severity ---
            st.markdown(
                f'<div class="pmc-card" style="border-left:6px solid {colour}">'
                f"<b>{result['equipment_id']}</b> — severity "
                f"<b style='color:{colour}'>{severity.upper()}</b>. "
                f"Last updated {result.get('last_updated','-')}.</div>",
                unsafe_allow_html=True,
            )

            # --- sensor summary ---
            st.subheader("Current Sensor Readings")
            sensors = result.get("sensor_summary", {})
            sensor_df = pd.DataFrame(
                {"sensor": list(sensors.keys()),
                 "value": list(sensors.values())}
            )
            st.dataframe(sensor_df, hide_index=True, use_container_width=True)

            # --- optional trend (nice-to-have, helps read direction) ---
            with st.expander("Show recent sensor trend (last 10 cycles)"):
                trend = get_recent_trend(eid, cycles=10)
                rows = trend.get("trend", [])
                if rows:
                    tdf = pd.DataFrame(rows).set_index("cycle")
                    st.line_chart(tdf[[c for c in tdf.columns if c.startswith("s")]])
                    st.caption("Drifting lines = degrading; flat lines = stable.")
                else:
                    st.info("No trend data available for this machine.")


# =========================================================================== #
# MODE 2 — Diagnostic Copilot
# =========================================================================== #
else:
    st.title("Diagnostic Copilot")
    st.caption("Paste a maintenance log or describe a symptom — the agent "
               "diagnoses the fault and grounds it in past incidents.")

    equipment_ids = ["(none)"] + load_equipment_ids()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        log_entry = st.text_area(
            "Maintenance log entry / symptom description",
            placeholder="e.g. DE bearing running hot at 82C, growl at 1480 rpm, "
                        "vibration trending up over 3 shifts.",
            height=120,
        )
    with col_b:
        eid_choice = st.selectbox("Equipment ID (optional)", equipment_ids)

    run = st.button("Run Diagnosis", type="primary")

    if run:
        if not log_entry.strip() and eid_choice == "(none)":
            st.warning("Provide a log entry and/or select an equipment ID.")
        else:
            equipment_id = None if eid_choice == "(none)" else eid_choice
            with st.spinner("Agent diagnosing — checking vitals, classifying, "
                            "retrieving similar incidents, self-reviewing…"):
                try:
                    # The agent requires equipment_id; if user gave none, use a
                    # placeholder so the anomaly tool still has something to look
                    # up (it will report 'unknown equipment' gracefully).
                    result = run_agent(
                        equipment_id=equipment_id or "(unspecified)",
                        log_entry=log_entry.strip() or None,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        "Diagnosis failed. This usually means AWS Bedrock "
                        "credentials or the Weaviate/MLflow services aren't "
                        f"available.\n\nDetails: `{exc}`"
                    )
                    result = None

            if result:
                hypothesis, action = split_hypothesis_action(
                    result.get("final_answer", "")
                )

                # --- headline metrics row ---
                m1, m2, m3 = st.columns(3)
                m1.metric("Fault Category Detected",
                          result.get("fault_category") or "—")
                conf = result.get("confidence_score", 0)
                m2.metric("Confidence Score", f"{conf}/10")
                m3.metric("Reasoning Iterations",
                          result.get("iterations_taken", 0))
                st.progress(min(max(conf / 10, 0.0), 1.0))

                st.markdown("---")

                # --- Similar Past Incidents ---
                st.markdown('<div class="pmc-section-title">Similar Past '
                            'Incidents</div>', unsafe_allow_html=True)
                n = result.get("similar_incidents_found", 0)
                if n:
                    st.success(f"Grounded in {n} similar historical incident(s).")
                else:
                    st.warning("No sufficiently similar past incident was found — "
                               "treat the hypothesis as unverified.")

                # --- Root Cause Hypothesis ---
                st.markdown('<div class="pmc-section-title">Root Cause '
                            'Hypothesis</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="pmc-card">{hypothesis or "—"}</div>',
                            unsafe_allow_html=True)

                # --- Recommended Action ---
                st.markdown('<div class="pmc-section-title">Recommended '
                            'Action</div>', unsafe_allow_html=True)
                action_text = action or (
                    "See hypothesis above; the agent did not separate a "
                    "distinct action."
                )
                st.markdown(f'<div class="pmc-card">{action_text}</div>',
                            unsafe_allow_html=True)
