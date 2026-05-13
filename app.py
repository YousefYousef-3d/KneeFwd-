import numpy as np
import pandas as pd
import streamlit as st


PRIMARY_BLUE = "#0b5cbf"

TRAJECTORIES = {
    "Persistent Low Function": "Low pre-operative function with minimal post-operative gain.",
    "Delayed Substantial Recovery": "Low pre-operative function followed by clinically meaningful recovery.",
    "Post-Operative Decline": "High pre-operative function followed by deterioration across recovery.",
    "Sustained High Function": "High pre-operative function preserved throughout the recovery year.",
}

MISSING_DATA_TYPES = [
    "Sleep (overnight non-wear)",
    "Charging",
    "Bathing / water exposure",
    "Sync / connectivity loss",
    "Battery depletion",
]

ACTIVITY_LEVELS = [
    "Sedentary",
    "Lightly active",
    "Moderately active",
    "Highly active",
]

ACTIVITY_LEVEL_DESCRIPTIONS = {
    "Sedentary": "Mostly seated or resting, with limited walking or standing time during the day.",
    "Lightly active": "Light household movement and short walks, roughly 2-4 active hours per day.",
    "Moderately active": "Regular walking, rehabilitation exercises, and daily movement, roughly 5-8 active hours per day.",
    "Highly active": "Frequent walking or rehabilitation activity, roughly 9-12 active hours per day.",
}

PAIN_SEVERITY_DESCRIPTIONS = {
    "Mild (0-3)": "Low pain burden on a 0-10 visual analogue scale.",
    "Moderate (4-6)": "Clinically meaningful pain that may limit activity and recovery pace.",
    "Severe (7-10)": "High pain burden likely to restrict mobility and daily function.",
}

MISSING_TYPE_DESCRIPTIONS = {
    "Sleep (overnight non-wear)": "Periods when the device is removed or poorly worn during sleep.",
    "Charging": "Data gaps created while the wearable is charging.",
    "Bathing / water exposure": "Device removal for showering, bathing, or water exposure.",
    "Sync / connectivity loss": "Observations lost because the device or phone fails to sync.",
    "Battery depletion": "Missing periods caused by the wearable battery running out.",
}


def inject_placeholder_missingness(
    dataframe: pd.DataFrame,
    missing_percent: int,
    mechanisms: list[str],
) -> pd.DataFrame:
    """Create visible gaps for interface preview only."""
    if missing_percent <= 0 or not mechanisms:
        return dataframe.copy()

    result = dataframe.copy()
    cluster_columns = [column for column in result.columns if column not in {"Day", "Date"}]
    mask_rng = np.random.default_rng(44)
    missing_probability = missing_percent / 100

    for column in cluster_columns:
        mask = mask_rng.random(len(result)) < missing_probability
        result.loc[mask, column] = np.nan

    return result


def make_placeholder_full_data() -> pd.DataFrame:
    """Build a 365-day scaffold so the output panels can be designed now."""
    days = np.arange(1, 366)
    dates = pd.date_range("2026-01-01", periods=365, freq="D")

    return pd.DataFrame(
        {
            "Day": days,
            "Date": dates,
            "Persistent Low Function": 36 + 2.0 * np.sin(days / 34),
            "Delayed Substantial Recovery": 32 + 0.17 * days + 2.5 * np.sin(days / 43),
            "Post-Operative Decline": 82 - 0.12 * days + 2.0 * np.cos(days / 39),
            "Sustained High Function": 80 + 1.8 * np.sin(days / 51),
        }
    )


def make_placeholder_prediction(missing_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Preview future imputed output without implementing a model."""
    predicted = missing_dataframe.copy()
    cluster_columns = [column for column in predicted.columns if column not in {"Day", "Date"}]
    predicted[cluster_columns] = (
        predicted[cluster_columns]
        .interpolate(limit_direction="both")
        .round(2)
    )
    return predicted


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def render_output_panel(title: str, dataframe: pd.DataFrame, key: str, caption: str) -> None:
    st.subheader(title)
    st.caption(caption)

    view = st.radio(
        "View format",
        ["Line chart", "Table (CSV)"],
        horizontal=True,
        key=f"{key}_view",
        label_visibility="collapsed",
    )

    chart_dataframe = dataframe.set_index("Date").drop(columns=["Day"], errors="ignore")
    if view == "Line chart":
        st.line_chart(chart_dataframe)
    else:
        st.dataframe(dataframe, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV",
        data=dataframe_to_csv(dataframe),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"{key}_download",
    )


def render_sidebar() -> dict:
    st.sidebar.header("Simulator Controls")

    st.sidebar.markdown("### 1. Cohort")
    patient_count = st.sidebar.number_input(
        "Number of simulated patients",
        min_value=10,
        max_value=5000,
        value=200,
        step=10,
        help="Total cohort size used by the simulator interface.",
    )
    st.sidebar.caption("Defines how many virtual patients belong to the simulated recovery cohort.")
    st.sidebar.markdown("---")

    average_bmi = st.sidebar.slider(
        "Average BMI",
        min_value=15.0,
        max_value=50.0,
        value=28.0,
        step=0.5,
        help="kg/m^2",
    )
    st.sidebar.caption("Mean body mass index for the simulated group, reported in kg/m^2.")
    st.sidebar.markdown("---")

    female_percent = st.sidebar.slider(
        "Proportion female (%)",
        min_value=0,
        max_value=100,
        value=55,
        step=1,
        help="Defines the sex distribution of the simulated cohort.",
    )
    st.sidebar.caption("The remaining percentage is treated as male for this interface-only version.")
    st.sidebar.markdown("---")

    pain_severity = st.sidebar.select_slider(
        "Pain severity (VAS band)",
        options=["Mild (0-3)", "Moderate (4-6)", "Severe (7-10)"],
        value="Moderate (4-6)",
        help="VAS means visual analogue scale, where 0 is no pain and 10 is worst imaginable pain.",
    )
    st.sidebar.caption(PAIN_SEVERITY_DESCRIPTIONS[pain_severity])
    st.sidebar.markdown("---")

    activity_level = st.sidebar.select_slider(
        "Baseline activity level",
        options=ACTIVITY_LEVELS,
        value="Moderately active",
        help="Approximate daily movement level before or near the start of recovery.",
    )
    st.sidebar.caption(ACTIVITY_LEVEL_DESCRIPTIONS[activity_level])

    st.sidebar.markdown("### 2. Recovery Model")
    trajectory = st.sidebar.selectbox(
        "Recovery trajectory (cluster)",
        options=list(TRAJECTORIES.keys()),
        index=1,
        help="Clinical recovery pattern assigned to the simulated group.",
    )
    st.sidebar.caption(TRAJECTORIES[trajectory])
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 3. Missing-Data Mechanism")
    randomize_missing_percent = st.sidebar.checkbox(
        "Randomize missing-data percentage",
        value=False,
        help="Draws a random missing-data percentage for the current simulation run.",
    )
    st.sidebar.caption("When enabled, the app samples a value from 0% to 100% for each new simulation run.")
    st.sidebar.markdown("---")

    manual_missing_percent = st.sidebar.slider(
        "Percentage of missing data",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        disabled=randomize_missing_percent,
        help="Approximate proportion of wearable observations removed from the complete dataset.",
    )
    st.sidebar.caption("For example, 20% means about one in five expected observations is missing.")
    st.sidebar.markdown("---")

    missing_types = st.sidebar.multiselect(
        "Type of missing data",
        options=MISSING_DATA_TYPES,
        default=MISSING_DATA_TYPES,
        help="Select the practical reasons wearable observations may be unavailable.",
    )
    selected_missing_descriptions = [
        f"- {missing_type}: {MISSING_TYPE_DESCRIPTIONS[missing_type]}"
        for missing_type in missing_types
    ]
    if selected_missing_descriptions:
        st.sidebar.caption("\n".join(selected_missing_descriptions))
    else:
        st.sidebar.caption("No missing-data mechanism selected.")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 4. Execute")
    run_clicked = st.sidebar.button("Run simulation", type="primary", use_container_width=True)
    st.sidebar.caption("Generates placeholder output panels for the selected interface settings.")
    if randomize_missing_percent:
        if run_clicked or "random_missing_percent" not in st.session_state:
            st.session_state["random_missing_percent"] = int(
                np.random.default_rng().integers(0, 101)
            )
        missing_percent = st.session_state["random_missing_percent"]
        st.sidebar.caption(f"Random draw for this run: {missing_percent}% missing data.")
    else:
        missing_percent = manual_missing_percent

    if run_clicked:
        st.session_state["ran"] = True

    return {
        "patient_count": int(patient_count),
        "average_bmi": average_bmi,
        "female_percent": female_percent,
        "pain_severity": pain_severity,
        "activity_level": activity_level,
        "trajectory": trajectory,
        "missing_percent": missing_percent,
        "missing_types": missing_types,
    }


def render_header() -> None:
    st.markdown(
        f"""
        <style>
            .block-container {{
                padding-top: 2.2rem;
                padding-bottom: 3rem;
            }}
            h1, h2, h3 {{
                font-family: Georgia, 'Times New Roman', serif;
            }}
            .kneefwd-title {{
                color: {PRIMARY_BLUE};
                font-family: Georgia, 'Times New Roman', serif;
                font-size: 3.2rem;
                font-weight: 700;
                letter-spacing: -0.04em;
                margin-bottom: 0.2rem;
            }}
            .kneefwd-subtitle {{
                color: #41516a;
                font-size: 1.05rem;
                margin-bottom: 1.8rem;
            }}
            .kneefwd-authors {{
                color: #2d4a6e;
                font-family: Georgia, 'Times New Roman', serif;
                font-size: 0.95rem;
                font-style: italic;
                margin-bottom: 0.75rem;
            }}
        </style>
        <div class="kneefwd-title">KneeFwd</div>
        <div class="kneefwd-authors">By: Behrad, Leena, Oli, Yuhe, Yousef</div>
        <div class="kneefwd-subtitle">
            An interface for simulating post-knee-replacement recovery trajectories and incomplete wearable-data observations.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cohort_summary(config: dict) -> None:
    female_count = round(config["patient_count"] * config["female_percent"] / 100)
    male_count = config["patient_count"] - female_count
    summary = pd.DataFrame(
        [
            {
                "Cluster": config["trajectory"],
                "Clinical label": TRAJECTORIES[config["trajectory"]],
                "Patients": config["patient_count"],
                "Female": female_count,
                "Male": male_count,
                "Average BMI": config["average_bmi"],
                "Pain severity": config["pain_severity"],
                "Activity level": config["activity_level"],
                "Missing data": f"{config['missing_percent']}%",
            }
        ]
    )

    st.subheader("Cohort Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_accuracy_panel() -> None:
    st.subheader("Section 4. Reconstruction Accuracy")
    metric_columns = st.columns(4)
    metric_columns[0].metric("RMSE", "--")
    metric_columns[1].metric("MAE", "--")
    metric_columns[2].metric("R^2", "--")
    metric_columns[3].metric("Coverage imputed", "--")
    st.caption("Metrics will populate once the imputation model is connected.")


def main() -> None:
    st.set_page_config(
        page_title="KneeFwd Simulator",
        page_icon="K",
        layout="wide",
    )

    config = render_sidebar()
    render_header()

    if not st.session_state.get("ran", False):
        st.info("Run simulation to view outputs.")
        return

    render_cohort_summary(config)

    full_data = make_placeholder_full_data()
    missing_data = inject_placeholder_missingness(
        full_data,
        config["missing_percent"],
        config["missing_types"],
    )
    predicted_data = make_placeholder_prediction(missing_data)

    render_output_panel(
        "Section 1. Simulated Full Data",
        full_data.round(2),
        "simulated_full_data",
        "Placeholder data only. Simulator logic has not yet been implemented.",
    )
    render_output_panel(
        "Section 2. Simulated Full Data With Missing Data",
        missing_data.round(2),
        "simulated_full_data_with_missing_data",
        "Placeholder missingness only. Mechanism-specific wearable data gaps will be connected later.",
    )
    render_output_panel(
        "Section 3. Missing Data Predicted Using Model",
        predicted_data.round(2),
        "missing_data_predicted_using_model",
        "Placeholder reconstruction only. The prediction model has not yet been implemented.",
    )
    render_accuracy_panel()


if __name__ == "__main__":
    main()
