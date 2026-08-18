import streamlit as st
import pandas as pd
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Urban Heat Emergency Response",
    page_icon="🌡️",
    layout="wide"
)

# =========================================================
# LOAD TEAM DATA
# =========================================================

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "zone.csv"

try:
    zones = pd.read_csv(DATA_FILE)
    zones.columns = zones.columns.str.strip().str.lower()
except Exception as e:
    st.error(f"Could not load zone.csv: {e}")
    st.stop()

# =========================================================
# TITLE
# =========================================================

st.title("🌡️ Urban Heat Emergency Response")

st.caption(
    "AI-influenced decision support for urban heatwave emergency management"
)

st.divider()

# =========================================================
# TOP SUMMARY
# =========================================================

critical_temp = (zones["temperature"] >= 42).sum()
high_vulnerability = (zones["vulnerability"] >= 75).sum()
low_cooling = (zones["cooling_access"] < 35).sum()
low_water = (zones["water_access"] < 35).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🔥 Extreme Heat Zones",
        critical_temp
    )

with col2:
    st.metric(
        "👥 Population Exposed",
        f"{zones['population'].sum():,}"
    )

with col3:
    st.metric(
        "🏥 High Vulnerability Zones",
        high_vulnerability
    )

with col4:
    st.metric(
        "🚰 Low Water Access",
        low_water
    )

# =========================================================
# ZONE OVERVIEW
# =========================================================

st.header("📍 Urban Heat Zone Assessment")

display_columns = [
    "zone",
    "temperature",
    "vegetation",
    "building_density",
    "population",
    "vulnerability",
    "cooling_access",
    "water_access",
    "hospital_capacity"
]

st.dataframe(
    zones[display_columns],
    use_container_width=True,
    hide_index=True
)

# =========================================================
# SELECT ZONE
# =========================================================

st.header("🔎 Zone-Specific Decision Support")

selected_zone = st.selectbox(
    "Select a zone to generate its response plan:",
    zones["zone"].tolist()
)

zone = zones[
    zones["zone"] == selected_zone
].iloc[0]

# =========================================================
# ZONE DETAILS
# =========================================================

st.subheader(f"📍 Zone {selected_zone}")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "🌡️ Temperature",
        f"{zone['temperature']:.1f} °C"
    )

with c2:
    st.metric(
        "🌳 Vegetation",
        f"{zone['vegetation']:.0f}%"
    )

with c3:
    st.metric(
        "🏢 Building Density",
        f"{zone['building_density']:.0f}%"
    )

with c4:
    st.metric(
        "👥 Population",
        f"{zone['population']:,.0f}"
    )

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "⚠️ Vulnerability",
        f"{zone['vulnerability']:.0f}%"
    )

with c2:
    st.metric(
        "❄️ Cooling Access",
        f"{zone['cooling_access']:.0f}%"
    )

with c3:
    st.metric(
        "🚰 Water Access",
        f"{zone['water_access']:.0f}%"
    )

with c4:
    st.metric(
        "🏥 Hospital Capacity",
        f"{zone['hospital_capacity']:.0f}%"
    )

# =========================================================
# CONTRIBUTING FACTORS
# =========================================================

st.subheader("🧠 Contributing Factors")

factors = []

if zone["temperature"] >= 42:
    factors.append("Extreme temperature")

if zone["vegetation"] < 20:
    factors.append("Very low vegetation")

if zone["building_density"] >= 80:
    factors.append("High building density")

if zone["vulnerability"] >= 75:
    factors.append("High population vulnerability")

if zone["cooling_access"] < 35:
    factors.append("Poor cooling access")

if zone["water_access"] < 35:
    factors.append("Poor water access")

if zone["hospital_capacity"] < 55:
    factors.append("Limited hospital capacity")

if factors:
    for factor in factors:
        st.write("•", factor)
else:
    st.success("No major contributing constraint detected.")

# =========================================================
# ZONE-SPECIFIC ACTION PLAN
# =========================================================

st.subheader("🚨 Recommended Actions")

actions = []

# Cooling
if zone["cooling_access"] < 35:
    actions.append(
        (
            "HIGH",
            "🏢 Deploy / open cooling centre",
            "Cooling access is critically low."
        )
    )

# Medical
if (
    zone["vulnerability"] >= 75
    or zone["hospital_capacity"] < 55
):
    actions.append(
        (
            "HIGH",
            "🏥 Deploy medical response team",
            "High vulnerability or limited hospital capacity."
        )
    )

# Water
if zone["water_access"] < 35:
    actions.append(
        (
            "HIGH",
            "🚰 Deploy emergency water tanker",
            "Water access is critically low."
        )
    )

# Heat warning
if zone["temperature"] >= 42:
    actions.append(
        (
            "HIGH",
            "📢 Issue targeted heat warning",
            "Extreme temperature detected."
        )
    )

# Temporary shade
if (
    zone["vegetation"] < 20
    and zone["building_density"] >= 80
):
    actions.append(
        (
            "MEDIUM",
            "⛱️ Deploy temporary shade/cooling",
            "Low vegetation combined with high built density."
        )
    )

# Monitoring
if not actions:
    actions.append(
        (
            "LOW",
            "👁️ Continue monitoring",
            "Current conditions do not require emergency deployment."
        )
    )

for number, (priority, action, reason) in enumerate(actions, 1):

    if priority == "HIGH":
        st.error(
            f"**{number}. {action}**  \n"
            f"Priority: **{priority}**  \n"
            f"Reason: {reason}"
        )

    elif priority == "MEDIUM":
        st.warning(
            f"**{number}. {action}**  \n"
            f"Priority: **{priority}**  \n"
            f"Reason: {reason}"
        )

    else:
        st.info(
            f"**{number}. {action}**  \n"
            f"Priority: **{priority}**  \n"
            f"Reason: {reason}"
        )

# =========================================================
# RESOURCE CAPACITY
# =========================================================

st.subheader("📦 Available Emergency Resources")

r1, r2, r3 = st.columns(3)

with r1:
    cooling_centres = st.number_input(
        "Cooling Centres Available",
        min_value=0,
        value=2
    )

with r2:
    medical_teams = st.number_input(
        "Medical Teams Available",
        min_value=0,
        value=2
    )

with r3:
    water_tankers = st.number_input(
        "Water Tankers Available",
        min_value=0,
        value=3
    )

# =========================================================
# RESOURCE CHECK
# =========================================================

st.subheader("📊 Resource Feasibility")

needs_cooling = zone["cooling_access"] < 35
needs_medical = (
    zone["vulnerability"] >= 75
    or zone["hospital_capacity"] < 55
)
needs_water = zone["water_access"] < 35

if needs_cooling:
    if cooling_centres > 0:
        st.success("✅ Cooling centre can be allocated.")
    else:
        st.error("❌ No cooling centre available.")

if needs_medical:
    if medical_teams > 0:
        st.success("✅ Medical team can be allocated.")
    else:
        st.error("❌ No medical team available.")

if needs_water:
    if water_tankers > 0:
        st.success("✅ Water tanker can be allocated.")
    else:
        st.error("❌ No water tanker available.")

# =========================================================
# HUMAN APPROVAL
# =========================================================

st.header("👤 Official Approval")

st.warning(
    "AI recommendations are advisory. "
    "Final action requires authorization by an official."
)

approval = st.radio(
    "Decision:",
    [
        "Pending",
        "Approve",
        "Reject",
        "Escalate"
    ],
    horizontal=True
)

if approval == "Approve":
    st.success(
        f"✅ Response plan for Zone {selected_zone} approved."
    )

elif approval == "Reject":
    st.error(
        f"❌ Response plan for Zone {selected_zone} rejected."
    )

elif approval == "Escalate":
    st.warning(
        f"⚠️ Zone {selected_zone} escalated for higher-level review."
    )

else:
    st.info("⏳ Waiting for official decision.")

# =========================================================
# REPLANNING
# =========================================================

st.header("🔄 Replanning")

st.write(
    "If temperature forecasts or service capacity change, "
    "the response plan should be recalculated."
)

if st.button("⚠️ Simulate Forecast / Capacity Change"):

    st.session_state["changed"] = True

if st.session_state.get("changed", False):

    st.error(
        "Conditions changed: the system detected a new situation."
    )

    st.info(
        "Replanning is required. The optimizer should "
        "recalculate zone priority and resource allocation."
    )

    if st.button("🔄 Recalculate Response Plan"):

        st.success(
            "New response plan generated for "
            f"Zone {selected_zone}."
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Prototype | AI-influenced decision support | "
    "Human approval required"
)