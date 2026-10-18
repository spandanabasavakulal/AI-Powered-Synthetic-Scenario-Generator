# AI-Powered Synthetic Scenario Generator

**UNSW-NB15 | Rare Attack Augmentation | Random Forest Model Comparison**

## Project Overview

The AI-Powered Synthetic Scenario Generator is a machine learning project designed to investigate whether synthetic data augmentation can improve the detection of rare cyberattacks in network intrusion detection systems.

The project uses the UNSW-NB15 dataset and focuses on the **Worms attack category**, which has significantly fewer training samples than common attack categories.

The project compares two Random Forest classifiers:

- **Real-only Random Forest:** Trained using the original training dataset.
- **Augmented Random Forest:** Trained using the original training dataset combined with 500 synthetic Worms samples.

The Streamlit dashboard displays the saved experimental results, model performance comparisons, confusion matrices, Worms-class evaluation metrics, and downloadable results.

## Objectives

- Analyze class imbalance in the UNSW-NB15 dataset.
- Generate additional candidate samples for the rare Worms attack category.
- Compare real-only and augmented Random Forest classifiers.
- Evaluate overall classification performance and rare-class detection.
- Visualize experimental results through an interactive Streamlit dashboard.

## Dataset

The project uses the UNSW-NB15 network intrusion detection dataset.

Required dataset files for the notebook experiments:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

The original dataset contains multiple network traffic categories, including Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, and Worms.

The original training dataset contains 130 Worms samples, while the testing dataset contains 44 Worms records.

## Methodology

### 1. Data Preparation

The training and testing datasets are prepared by removing the identifier column and separating input features from the target labels. Categorical features are encoded, and missing values are handled during model preprocessing.

### 2. Synthetic Sample Generation

The project uses **perturbation-based sampling** to generate 500 additional candidate Worms samples.

The method resamples existing Worms records and applies controlled Gaussian noise to selected numerical features, keeping perturbed values within the observed ranges of the original Worms samples.

CTGAN was explored during experimentation, but its generated samples were not used in the final model comparison because they did not pass the selected validation checks.

### 3. Dataset Augmentation

Two datasets are prepared:

- **Real-only dataset:** Original training records.
- **Augmented dataset:** Original training records combined with 500 synthetic Worms samples.

The Worms training count increases from 130 to 630 after augmentation.

### 4. Model Training and Evaluation

Two Random Forest classifiers are trained using the same preprocessing approach. Both models are evaluated on the original, unchanged test dataset.

The evaluation includes:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrices
- Worms-class precision, recall, and F1-score
- True positives, false negatives, and false positives for Worms detection

### 5. Streamlit Dashboard

The dashboard presents saved results from the Jupyter Notebook. It does not retrain the models when opened.

The dashboard includes the following sections:

- **Dashboard:** Experiment overview, overall model performance, and attack category distribution.
- **Synthetic Samples:** Displays the generated Worms samples.
- **Model Evaluation:** Compares overall model performance, confusion matrices, and Worms-class detection metrics.
- **Download Results:** Provides available saved experiment outputs for download.

## Experimental Results

### Overall Model Performance

| Metric | Real-only Random Forest | Augmented Random Forest |
|---|---:|---:|
| Accuracy | 74.96% | 74.97% |
| Weighted Precision | 83.73% | 83.79% |
| Weighted Recall | 74.96% | 74.97% |
| Weighted F1-score | 78.35% | 78.40% |

### Worms-Class Detection

| Metric | Real-only | Augmented |
|---|---:|---:|
| Precision | 58.33% | 73.33% |
| Recall | 15.91% | 25.00% |
| F1-score | 25.00% | 37.29% |
| True Positives | 7 | 11 |
| False Negatives | 37 | 33 |
| False Positives | 5 | 4 |

The experimental results show a small improvement in overall performance and improved Worms-class detection after augmentation. However, the augmented model still misses some Worms attacks, and the rare-class evaluation is based on only 44 actual Worms test records. Further validation is required to assess generalization.

## Project Structure

```text
AI_Synthetic_Scenario_Generator_Streamlit/
│
├── app.py
├── requirements.txt
├── README.md
│
└── exported_results/
    ├── evaluation_predictions.csv
    ├── overall_metrics.csv
    ├── synthetic_worms.csv
    ├── real_model.joblib
    └── augmented_model.joblib
```

The dashboard requires the saved CSV files shown above. The two `.joblib` model files are retained as exported experiment artifacts; the current dashboard displays saved results rather than loading these models for new predictions.

## Requirements

- Python 3.10 or newer
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

Install the dependencies listed in `requirements.txt`.

## Run Locally

1. Open a terminal in the project folder containing `app.py`.

2. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the Streamlit dashboard:

   ```bash
   python -m streamlit run app.py
   ```

4. Open the local URL displayed in the terminal, usually:

   ```text
   http://localhost:8501
   ```

5. Explore the Dashboard, Synthetic Samples, Model Evaluation, and Download Results sections.

**Note:** The current dashboard reads saved experiment outputs from the `exported_results` folder. The original training and testing CSV files are needed to reproduce the notebook experiments, but they do not need to be uploaded to view the saved results.

## Limitations

- The generated samples are perturbation-based candidates derived from existing Worms records; they are not independent real-world attack observations.
- CTGAN-generated samples were not used in the final experiment.
- Overall performance improved only slightly.
- Worms-class evaluation uses a small test subset of 44 records.
- Additional experiments on independent datasets and real-world network traffic are needed before drawing broader conclusions.

## Conclusion

This project investigates the use of perturbation-based synthetic data augmentation to improve rare cyberattack detection using the UNSW-NB15 dataset.

The augmented Random Forest achieved a Worms recall of 25.00%, compared with 15.91% for the real-only model. The results suggest that augmentation may help detect additional rare attacks in this experiment, although further validation is necessary.

The Streamlit dashboard provides a visual interface for examining and comparing the saved experimental results.