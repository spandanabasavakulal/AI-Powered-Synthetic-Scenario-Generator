
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Powered Synthetic Scenario Generator",
    page_icon="🛡️",
    layout="wide",
)

st.title("AI-Powered Synthetic Scenario Generator")

st.caption(
    "UNSW-NB15 | Rare Attack Augmentation | "
    "Random Forest Model Comparison"
)

st.info(
    "This dashboard displays saved results from your Jupyter Notebook. "
    "It does not retrain your models. The final augmentation method uses "
    "perturbation-based sampling. CTGAN-generated samples were not used "
    "because they failed the selected validation checks."
)

# =========================================================
# 2. LOAD EXPORTED RESULTS
# =========================================================

RESULTS_DIR = Path(__file__).parent / "exported_results"

PREDICTIONS_FILE = RESULTS_DIR / "evaluation_predictions.csv"
SYNTHETIC_FILE = RESULTS_DIR / "synthetic_worms.csv"
METRICS_FILE = RESULTS_DIR / "overall_metrics.csv"

required_files = [
    PREDICTIONS_FILE,
    SYNTHETIC_FILE,
    METRICS_FILE,
]

missing_files = [
    file.name for file in required_files if not file.exists()
]

if missing_files:
    st.error(
        "The following files are missing from exported_results: "
        + ", ".join(missing_files)
    )

    st.code(
        "project_folder/\n"
        "├── app.py\n"
        "└── exported_results/\n"
        "    ├── evaluation_predictions.csv\n"
        "    ├── synthetic_worms.csv\n"
        "    └── overall_metrics.csv"
    )

    st.stop()


@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)


try:
    predictions_df = load_csv(str(PREDICTIONS_FILE))
    synthetic_df = load_csv(str(SYNTHETIC_FILE))
    saved_metrics_df = load_csv(str(METRICS_FILE))

except Exception as error:
    st.error(f"Could not load the exported results: {error}")
    st.stop()


# =========================================================
# 3. VALIDATE PREDICTION FILE
# =========================================================

required_prediction_columns = [
    "actual",
    "real_only_prediction",
    "augmented_prediction",
]

if not all(
    column in predictions_df.columns
    for column in required_prediction_columns
):
    st.error(
        "evaluation_predictions.csv must contain these columns: "
        "actual, real_only_prediction, augmented_prediction."
    )
    st.stop()

if predictions_df.empty:
    st.error("The evaluation predictions file contains no records.")
    st.stop()

if predictions_df[required_prediction_columns].isna().any().any():
    st.error(
        "The evaluation predictions contain missing values. "
        "Please re-export the predictions from your notebook."
    )
    st.stop()

actual = predictions_df["actual"].astype(str)
real_predictions = predictions_df[
    "real_only_prediction"
].astype(str)
augmented_predictions = predictions_df[
    "augmented_prediction"
].astype(str)


# =========================================================
# 4. CALCULATE OVERALL METRICS
# =========================================================

def calculate_metrics(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "Weighted Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "Weighted F1-score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


metrics_df = pd.DataFrame([
    calculate_metrics(
        actual,
        real_predictions,
        "Real-only Random Forest",
    ),
    calculate_metrics(
        actual,
        augmented_predictions,
        "Augmented Random Forest",
    ),
])


# =========================================================
# 5. CALCULATE WORMS-CLASS METRICS
# =========================================================

worms_results = []

for model_name, model_predictions in [
    ("Real-only", real_predictions),
    ("Augmented", augmented_predictions),
]:
    true_positive = int(
        (
            (actual == "Worms")
            & (model_predictions == "Worms")
        ).sum()
    )

    false_negative = int(
        (
            (actual == "Worms")
            & (model_predictions != "Worms")
        ).sum()
    )

    false_positive = int(
        (
            (actual != "Worms")
            & (model_predictions == "Worms")
        ).sum()
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    worms_results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "True Positives": true_positive,
        "False Negatives": false_negative,
        "False Positives": false_positive,
    })

worms_metrics_df = pd.DataFrame(worms_results)


# =========================================================
# 6. HELPER FUNCTION: GROUPED BAR CHART
# =========================================================

def create_grouped_bar_chart(
    labels,
    first_values,
    second_values,
    first_name,
    second_name,
    title,
    y_label="Score",
    percentage=False,
):
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(
        x - width / 2,
        first_values,
        width,
        label=first_name,
    )

    bars2 = ax.bar(
        x + width / 2,
        second_values,
        width,
        label=second_name,
    )

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    if percentage:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )
        ax.bar_label(
            bars1,
            fmt="%.1f%%",
            labels=[f"{value * 100:.1f}%" for value in first_values],
            padding=3,
            fontsize=8,
        )
        ax.bar_label(
            bars2,
            labels=[f"{value * 100:.1f}%" for value in second_values],
            padding=3,
            fontsize=8,
        )
    else:
        max_value = max(
            list(first_values) + list(second_values) + [0.1]
        )
        ax.set_ylim(0, max_value * 1.2)

        ax.bar_label(bars1, fmt="%.0f", padding=3, fontsize=8)
        ax.bar_label(bars2, fmt="%.0f", padding=3, fontsize=8)

    fig.tight_layout()
    return fig


# =========================================================
# 7. DASHBOARD TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Synthetic Samples",
    "Model Evaluation",
    "Download Results",
])


# =========================================================
# TAB 1: DASHBOARD
# =========================================================

with tab1:
    st.header("Experiment Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Records",
        f"{len(predictions_df):,}",
    )

    col2.metric(
        "Synthetic Samples",
        f"{len(synthetic_df):,}",
    )

    # Original training dataset contains 130 Worms records.
    # Change this value if your training dataset changes.
    original_worms_count = 130

    col3.metric(
        "Original Worms Samples",
        f"{original_worms_count:,}",
    )

    col4.metric(
        "Worms After Augmentation",
        f"{original_worms_count + len(synthetic_df):,}",
    )

    st.divider()

    st.subheader("Overall Model Performance")

    display_metrics = metrics_df.copy()

    metric_columns = [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1-score",
    ]

    st.dataframe(
        display_metrics.style.format({
            column: "{:.4f}" for column in metric_columns
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Overall Performance Comparison")

    fig = create_grouped_bar_chart(
        labels=[
            "Accuracy",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1-score",
        ],
        first_values=[
            metrics_df.iloc[0]["Accuracy"],
            metrics_df.iloc[0]["Weighted Precision"],
            metrics_df.iloc[0]["Weighted Recall"],
            metrics_df.iloc[0]["Weighted F1-score"],
        ],
        second_values=[
            metrics_df.iloc[1]["Accuracy"],
            metrics_df.iloc[1]["Weighted Precision"],
            metrics_df.iloc[1]["Weighted Recall"],
            metrics_df.iloc[1]["Weighted F1-score"],
        ],
        first_name="Real-only Random Forest",
        second_name="Augmented Random Forest",
        title="Real-only vs Augmented Random Forest",
        percentage=True,
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Attack Category Distribution")

    st.write(
        "Optionally upload the original UNSW-NB15 training CSV "
        "to visualize the distribution of attack categories."
    )

    train_file = st.file_uploader(
        "Upload UNSW-NB15 training CSV",
        type=["csv"],
        key="training_distribution",
    )

    if train_file is not None:
        try:
            training_distribution = pd.read_csv(
                train_file,
                low_memory=False,
            )

            if "attack_cat" not in training_distribution.columns:
                st.error(
                    "The uploaded file does not contain "
                    "the attack_cat column."
                )
            else:
                counts = (
                    training_distribution["attack_cat"]
                    .fillna("Unknown")
                    .value_counts()
                )

                fig, ax = plt.subplots(figsize=(12, 5))

                counts.plot(
                    kind="bar",
                    ax=ax,
                )

                ax.set_title(
                    "Attack Category Distribution — Training Data"
                )
                ax.set_xlabel("Attack Category")
                ax.set_ylabel("Number of Records")
                ax.tick_params(
                    axis="x",
                    labelrotation=45,
                )

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        except Exception as error:
            st.error(f"Could not read the uploaded CSV: {error}")


# =========================================================
# TAB 2: SYNTHETIC SAMPLES
# =========================================================

with tab2:
    st.header("Synthetic Worms Samples")

    st.caption(
        "These records were generated by resampling real Worms "
        "records and applying small perturbations to selected "
        "continuous features. They are not CTGAN-generated samples."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Generated Records",
        f"{len(synthetic_df):,}",
    )

    col2.metric(
        "Feature Columns",
        f"{synthetic_df.shape[1]:,}",
    )

    st.subheader("Synthetic Sample Preview")

    if synthetic_df.empty:
        st.warning("No synthetic samples were found in the CSV.")
    else:
        st.dataframe(
            synthetic_df.head(30),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Basic Validation")

    missing_values = int(
        synthetic_df.isna().sum().sum()
    )

    st.metric(
        "Missing Values",
        missing_values,
    )

    continuous_columns = [
        "dur",
        "rate",
        "sload",
        "dload",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "tcprtt",
        "synack",
        "ackdat",
    ]

    available_continuous = [
        column
        for column in continuous_columns
        if column in synthetic_df.columns
    ]

    if available_continuous:
        numeric_data = synthetic_df[
            available_continuous
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        negative_counts = (
            numeric_data.lt(0)
            .sum()
            .rename("Negative Values")
            .to_frame()
        )

        st.write(
            "Negative values in selected continuous features"
        )

        st.dataframe(
            negative_counts,
            use_container_width=True,
        )

        non_numeric_or_missing = int(
            numeric_data.isna().sum().sum()
        )

        if non_numeric_or_missing > 0:
            st.warning(
                f"{non_numeric_or_missing} selected numeric feature "
                "values could not be interpreted as numbers."
            )

    st.warning(
        "Passing basic validation checks does not prove that synthetic "
        "records are realistic or represent previously unseen attacks. "
        "Further statistical and downstream model validation is needed."
    )


# =========================================================
# TAB 3: MODEL EVALUATION
# =========================================================

with tab3:
    st.header("Model Evaluation")

    st.subheader("Overall Performance Comparison")

    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.4f}",
            "Weighted Precision": "{:.4f}",
            "Weighted Recall": "{:.4f}",
            "Weighted F1-score": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Corrected grouped chart: bars are side by side.
    fig = create_grouped_bar_chart(
        labels=[
            "Accuracy",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1-score",
        ],
        first_values=[
            metrics_df.iloc[0]["Accuracy"],
            metrics_df.iloc[0]["Weighted Precision"],
            metrics_df.iloc[0]["Weighted Recall"],
            metrics_df.iloc[0]["Weighted F1-score"],
        ],
        second_values=[
            metrics_df.iloc[1]["Accuracy"],
            metrics_df.iloc[1]["Weighted Precision"],
            metrics_df.iloc[1]["Weighted Recall"],
            metrics_df.iloc[1]["Weighted F1-score"],
        ],
        first_name="Real-only Random Forest",
        second_name="Augmented Random Forest",
        title="Overall Performance Comparison",
        percentage=True,
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # -----------------------------------------------------
    # CONFUSION MATRICES
    # -----------------------------------------------------

    st.subheader("Confusion Matrices")

    labels = sorted(
        set(actual.unique())
        | set(real_predictions.unique())
        | set(augmented_predictions.unique())
    )

    left, right = st.columns(2)

    models_to_display = [
        (
            left,
            "Real-only Random Forest",
            real_predictions,
        ),
        (
            right,
            "Augmented Random Forest",
            augmented_predictions,
        ),
    ]

    for column, model_name, model_predictions in models_to_display:
        cm = confusion_matrix(
            actual,
            model_predictions,
            labels=labels,
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels,
        )

        display.plot(
            ax=ax,
            xticks_rotation=90,
            values_format="d",
            colorbar=False,
        )

        ax.set_title(model_name)
        ax.set_xlabel("Predicted Category")
        ax.set_ylabel("Actual Category")

        fig.tight_layout()
        column.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # -----------------------------------------------------
    # WORMS ATTACK DETECTION
    # -----------------------------------------------------

    st.subheader("Worms Attack Detection")

    st.dataframe(
        worms_metrics_df.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-score": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = create_grouped_bar_chart(
        labels=["Precision", "Recall", "F1-score"],
        first_values=[
            worms_metrics_df.iloc[0]["Precision"],
            worms_metrics_df.iloc[0]["Recall"],
            worms_metrics_df.iloc[0]["F1-score"],
        ],
        second_values=[
            worms_metrics_df.iloc[1]["Precision"],
            worms_metrics_df.iloc[1]["Recall"],
            worms_metrics_df.iloc[1]["F1-score"],
        ],
        first_name="Real-only",
        second_name="Augmented",
        title="Worms-Class Performance Comparison",
        percentage=True,
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    worms_test_count = int((actual == "Worms").sum())

    st.caption(
        f"The evaluation set contains {worms_test_count} actual Worms "
        "records. Interpret class-specific metrics cautiously because "
        "the rare-class test sample is small."
    )

    st.info(
        "The dashboard evaluates saved predictions. It does not train "
        "models or establish that augmentation alone caused the observed "
        "performance differences."
    )


# =========================================================
# TAB 4: DOWNLOAD RESULTS
# =========================================================

with tab4:
    st.header("Download Results")

    st.write(
        "Download the exported synthetic samples and model evaluation "
        "results for reporting and further analysis."
    )

    st.download_button(
        label="Download Synthetic Worms Samples",
        data=synthetic_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="synthetic_worms_perturbation_samples.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Model Comparison Metrics",
        data=metrics_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="model_comparison_metrics.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Evaluation Predictions",
        data=predictions_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="evaluation_predictions.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Worms Detection Metrics",
        data=worms_metrics_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="worms_detection_metrics.csv",
        mime="text/csv",
    )

    st.caption(
        "All evaluation metrics are calculated from the saved notebook "
        "predictions. No model training is performed in this dashboard."
    )
