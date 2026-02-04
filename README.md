Here's a comprehensive `README.md` file for your Streamlit application lab project, designed for clarity, professionalism, and ease of use.

---

# QuLab: Lab 4: Data Quality, Provenance & Bias Metrics Dashboard

![Quant University Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Description

Welcome to **QuLab: Lab 4**, a hands-on Streamlit application designed for ML Engineers, Model Validators, and Risk/Compliance Partners. This dashboard empowers users to perform comprehensive data quality, provenance, and bias assessments on datasets before they are used for model training or deployment.

**Scenario**: Step into the shoes of Maya, a Senior ML Engineer at "Software Innovators Inc.". Maya's team is tasked with developing robust machine learning models that adhere to strict regulatory standards and deliver fair outcomes. Before committing to costly model training, Maya needs to ensure the raw data meets fundamental quality standards, does not contain hidden biases, and has not drifted significantly from historical data. This application helps Maya achieve her goal, preventing unnecessary model risk and ensuring compliance.

The application provides a structured workflow for:
1.  **Loading and Configuring Datasets**: Setting up the primary and optional baseline datasets with assessment parameters.
2.  **Assessing Data Quality**: Identifying common data issues like missing values, duplicates, and inconsistencies.
3.  **Quantifying Data Bias**: Measuring inherent biases across sensitive attributes in the raw data.
4.  **Detecting Data Drift**: Comparing new data distributions against a historical baseline.
5.  **Generating Comprehensive Reports**: Synthesizing findings into an overall readiness decision and providing auditable artifacts.

By proactively identifying and addressing these data-related risks, users can ensure their ML models are built on a foundation of high-quality, fair, and stable data.

## Features

This application offers a guided, multi-step process for data risk assessment:

*   **Data Upload & Configuration**:
    *   Securely upload primary and optional baseline datasets (CSV format).
    *   Interactively select target label column and sensitive attributes for assessment.
    *   Define privileged and unprivileged groups for each sensitive attribute.
    *   Customize or override default warning and failure thresholds for Data Quality, Bias, and Drift metrics to align with specific organizational risk appetites.
    *   Set feature-specific range expectations for numerical columns.
*   **Data Quality Assessment**:
    *   Automated checks for core data quality dimensions:
        *   **Missingness Ratio**: Proportion of missing values per feature.
        *   **Duplicate Rows Ratio**: Overall percentage of duplicate rows in the dataset.
        *   **Type Inconsistency**: Ratio of values not conforming to the inferred data type.
        *   **Range Violation Ratio**: Percentage of numerical values falling outside defined expected ranges.
        *   **Cardinality Check**: Analysis of unique values for categorical features (too low or too high).
    *   Provides feature-level and overall dataset quality status (PASS, WARN, FAIL).
*   **Bias Metrics Computation**:
    *   Quantifies inherent biases in the raw data across specified sensitive attributes and protected groups.
    *   Calculates key pre-training bias metrics:
        *   **Demographic Parity Difference (DPD)**: Difference in favorable outcome rates.
        *   **Disparate Impact Ratio (DIR)**: Ratio of favorable outcome rates.
        *   **Proxy True Positive Rate Gap (TPR Gap)**: Difference in actual positive outcome rates.
        *   **Proxy False Positive Rate Gap (FPR Gap)**: Difference in actual negative outcome rates.
    *   Assigns status (PASS, WARN, FAIL) based on configured thresholds.
*   **Drift Detection**:
    *   Compares the primary dataset against an optional baseline dataset.
    *   Calculates **Population Stability Index (PSI)** for numerical features to quantify distribution shifts.
    *   Visualizes feature distributions (histograms) for primary and baseline data to highlight drift.
    *   Provides feature-level and overall drift status.
*   **Summary & Export**:
    *   Generates an **Overall Dataset Readiness Decision** (DO NOT DEPLOY, PROCEED WITH MITIGATION, PROCEED) based on aggregated assessment results.
    *   Creates a comprehensive audit bundle (ZIP archive) containing:
        *   Executive Summary (Markdown).
        *   Detailed JSON reports for Data Quality, Bias, and Drift.
        *   Snapshot of the assessment configuration.
        *   Original primary and baseline datasets (for provenance).
        *   An `evidence_manifest.json` with SHA-256 hashes of all artifacts for tamper-evident auditability.
    *   Allows direct download of the audit bundle from the dashboard.
*   **Interactive Streamlit UI**: Intuitive, step-by-step user interface with clear instructions and real-time feedback.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
    (Replace `your-username/your-repository-name` with the actual repository path if available.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install the required packages:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    numpy>=1.20.0
    matplotlib>=3.5.0
    seaborn>=0.11.0
    # Add any other dependencies if source.py uses them, e.g.:
    # scikit-learn
    # scipy
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is active and you are in the project's root directory.
    ```bash
    streamlit run app.py
    ```

2.  **Access the Dashboard:**
    Your web browser should automatically open to the Streamlit application (usually at `http://localhost:8501`).

3.  **Follow the Assessment Steps:**
    *   **Data Upload & Configuration**: Upload your primary (and optional baseline) CSV datasets. Select the target column, sensitive attributes, define protected groups, and review/override default thresholds. Click "Apply Configuration".
    *   **Data Quality**: Click "Perform Data Quality Checks" to analyze the primary dataset for common issues.
    *   **Bias Metrics**: Click "Compute Bias Metrics" to assess fairness across protected groups.
    *   **Drift**: If a baseline dataset was provided, click "Detect Data Drift (PSI)" to compare distributions.
    *   **Summary & Export**: Review the overall readiness decision. Click "Generate Final Report and Export Artifacts" to create and download a comprehensive audit bundle.

The sidebar provides a navigation menu and a real-time status update of the assessment progress.

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains core logic for data quality, bias, drift, and reporting functions
├── requirements.txt        # List of Python dependencies
├── .streamlit/             # Streamlit configuration directory (optional)
│   └── config.toml         # Streamlit specific configurations (e.g., theme)
├── temp_data/              # Temporary directory for uploaded CSV files (created on demand)
├── data/                   # Directory for storing generated raw data artifacts (created on demand)
├── reports/                # Directory for storing generated assessment reports (created on demand)
└── README.md               # This README file
```

## Technology Stack

*   **Python**: Programming language
*   **Streamlit**: For creating interactive web applications
*   **Pandas**: For data manipulation and analysis
*   **NumPy**: For numerical operations
*   **Matplotlib**: For static visualizations
*   **Seaborn**: For enhanced statistical data visualizations
*   **JSON**: For structured data output and configuration
*   **OS/Datetime/Zipfile/Hashlib**: Standard Python libraries for file system operations, timestamping, archiving, and cryptographic hashing.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and ensure tests pass (if applicable).
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You might need to create a `LICENSE` file in your repository).

## Contact

For questions, feedback, or collaborations, please refer to:

*   **Quant University Website**: [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com (Example, update as needed)

---