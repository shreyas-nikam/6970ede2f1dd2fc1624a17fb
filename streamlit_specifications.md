
# Streamlit Application Specification: Data Quality, Provenance & Bias Metrics Dashboard

## 1. Application Overview

The **Data Quality, Provenance & Bias Metrics Dashboard** is a Streamlit application designed to provide an objective, quantitative assessment of data risk for supervised Machine Learning (ML) systems. It serves as a crucial tool for ML Engineers, Model Validators, and Risk/Compliance Partners to determine if a dataset is fit for model training, validation, or deployment.

**High-Level Story Flow:**

1.  **Welcome & Context:** The application introduces the persona of Maya, a Senior ML Engineer at "Software Innovators Inc.", tasked with assessing a new dataset for a credit approval model.
2.  **Data Upload & Configuration:** Maya uploads her primary credit application dataset and an optional historical baseline. She then configures critical parameters: the target label, sensitive attributes, protected group definitions, and custom thresholds for data quality, bias, and drift metrics.
3.  **Data Quality Assessment:** The app guides Maya through a series of data quality checks (missingness, duplicates, type consistency, range violations, cardinality) for each feature. She reviews the results, noting 'PASS', 'WARN', or 'FAIL' statuses, which helps her identify fundamental issues requiring immediate attention.
4.  **Bias Metrics Computation:** Maya proceeds to quantify potential biases in the dataset, examining metrics like Demographic Parity Difference and Disparate Impact Ratio across sensitive attributes. This helps Software Innovators Inc. ensure fair lending practices and comply with ethical guidelines.
5.  **Drift Detection:** If a baseline dataset was provided, the application computes the Population Stability Index (PSI) for numerical features, allowing Maya to detect significant shifts in data distributions compared to historical data. Visualizations aid in understanding these shifts.
6.  **Summary & Export:** Finally, all assessment results are synthesized into an overall dataset readiness decision ('DO NOT DEPLOY', 'PROCEED WITH MITIGATION', or 'PROCEED'). The application generates comprehensive JSON reports, a human-readable executive summary, and bundles all audit-ready artifacts, including an SHA-256 evidence manifest, into a secure zip archive. This ensures traceability and compliance for internal review and regulatory examinations.

## 2. Code Requirements

### Import Statements

```python
import streamlit as st
import pandas as pd
import json
import os
import datetime
import zipfile
import hashlib
import numpy as np # For numpy operations within Streamlit context if needed, e.g., for `pd.api.types.is_numeric_dtype`
import matplotlib.pyplot as plt # For drift visualizations
import seaborn as sns # For drift visualizations

# Import all functions from the provided source.py file
from source import (
    generate_sample_data, # Only used for initial setup in source.py, not directly called in app flow
    load_and_configure_datasets,
    perform_data_quality_checks,
    compute_bias_metrics,
    calculate_psi,
    make_readiness_decision,
    generate_reports,
    create_evidence_manifest,
    bundle_artifacts
)
```

### Streamlit Application Structure and Flow

The application will simulate a multi-page experience using `st.selectbox` in the sidebar for navigation.

#### Session State Design (`st.session_state`)

The following `st.session_state` keys will be used to preserve state across interactions:

*   **Initialization:**
    ```python
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Data Upload & Configuration"
    if 'primary_df' not in st.session_state:
        st.session_state['primary_df'] = None
    if 'baseline_df' not in st.session_state:
        st.session_state['baseline_df'] = None
    if 'assessment_config' not in st.session_state:
        st.session_state['assessment_config'] = None
    if 'data_quality_results' not in st.session_state:
        st.session_state['data_quality_results'] = None
    if 'bias_metrics_results' not in st.session_state:
        st.session_state['bias_metrics_results'] = None
    if 'drift_detection_results' not in st.session_state:
        # Default if no baseline is provided, will be overwritten if baseline is uploaded
        st.session_state['drift_detection_results'] = {'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
    if 'readiness_decision' not in st.session_state:
        st.session_state['readiness_decision'] = 'Pending Configuration'
    if 'reports_folder_path' not in st.session_state:
        st.session_state['reports_folder_path'] = None
    if 'zip_archive_path' not in st.session_state:
        st.session_state['zip_archive_path'] = None
    if 'primary_data_filename' not in st.session_state:
        st.session_state['primary_data_filename'] = None
    if 'baseline_data_filename' not in st.session_state:
        st.session_state['baseline_data_filename'] = None
    if 'primary_temp_path' not in st.session_state:
        st.session_state['primary_temp_path'] = None
    if 'baseline_temp_path' not in st.session_state:
        st.session_state['baseline_temp_path'] = None
    if 'config_applied' not in st.session_state:
        st.session_state['config_applied'] = False
    ```

*   **Updates:**
    *   `st.session_state['primary_df']`, `st.session_state['baseline_df']`: Updated after successful file upload and parsing.
    *   `st.session_state['assessment_config']`: Updated after "Apply Configuration" button is clicked.
    *   `st.session_state['data_quality_results']`, `st.session_state['bias_metrics_results']`, `st.session_state['drift_detection_results']`: Updated when their respective analysis functions are called.
    *   `st.session_state['readiness_decision']`: Updated on the "Summary & Export" page after `make_readiness_decision` is called.
    *   `st.session_state['reports_folder_path']`, `st.session_state['zip_archive_path']`: Updated on the "Summary & Export" page after report generation and bundling.
    *   `st.session_state['primary_data_filename']`, `st.session_state['baseline_data_filename']`: Store the names of uploaded files.
    *   `st.session_state['primary_temp_path']`, `st.session_state['baseline_temp_path']`: Store temporary file paths where uploaded files are saved.
    *   `st.session_state['config_applied']`: Set to `True` once configuration is successfully applied.

*   **Reads:** All `st.session_state` keys are read across different "pages" (conditional rendering blocks) to maintain context and pass data between steps.

#### Helper function to save uploaded files temporarily

```python
def _save_uploaded_file(uploaded_file, directory="temp_data"):
    if uploaded_file is not None:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None
```

#### Sidebar Navigation

```python
with st.sidebar:
    st.markdown(f"# Data Risk Assessment")
    st.markdown(f"**Persona:** Maya, Senior ML Engineer")
    st.markdown(f"---")
    st.session_state['current_page'] = st.selectbox(
        "Choose an Assessment Step:",
        ["Data Upload & Configuration", "Data Quality", "Bias Metrics", "Drift", "Summary & Export"],
        key="page_selector"
    )
    st.markdown(f"---")
    st.markdown(f"## Current Status:")
    if st.session_state.get('config_applied', False):
        st.markdown(f"✅ Configuration Applied")
    else:
        st.markdown(f"⏳ Awaiting Configuration")
    if st.session_state.get('data_quality_results') is not None:
        st.markdown(f"✅ Data Quality: **{st.session_state['data_quality_results']['overall_dataset_quality_status']}**")
    else:
        st.markdown(f"⏳ Data Quality Pending")
    if st.session_state.get('bias_metrics_results') is not None:
        st.markdown(f"✅ Bias Metrics: **{st.session_state['bias_metrics_results']['overall_bias_status']}**")
    else:
        st.markdown(f"⏳ Bias Metrics Pending")
    if st.session_state.get('baseline_df') is not None and st.session_state.get('drift_detection_results') is not None:
        st.markdown(f"✅ Drift: **{st.session_state['drift_detection_results']['overall_drift_status']}**")
    elif st.session_state.get('baseline_df') is None and st.session_state.get('config_applied'):
        st.markdown(f"⚠️ Drift: No Baseline Provided")
    else:
        st.markdown(f"⏳ Drift Pending")
    if st.session_state.get('readiness_decision') != 'Pending Configuration':
        st.markdown(f"**Final Decision:** **{st.session_state['readiness_decision']}**")
    else:
        st.markdown(f"⏳ Final Decision Pending")

```

#### Page: Data Upload & Configuration

```python
if st.session_state['current_page'] == "Data Upload & Configuration":
    st.markdown(f"# 1. Setting the Stage: Data Loading and Configuration")
    st.markdown(f"Welcome to this hands-on lab designed for ML Engineers, Model Validators, and Risk/Compliance Partners. In this notebook, we will step into the shoes of Maya, a Senior ML Engineer at \"Software Innovators Inc.\". Maya's team is responsible for developing and maintaining robust machine learning models that adhere to strict regulatory standards and deliver fair outcomes to customers.")
    st.markdown(f"Today, Maya is tasked with preparing a new dataset for a credit approval model. Before Software Innovators Inc. commits to costly model training and potential deployment, Maya needs to perform a comprehensive data quality and risk assessment. Her goal is to ensure the raw data meets fundamental quality standards, does not contain hidden biases, and has not drifted significantly from historical data, thereby preventing unnecessary model risk and ensuring compliance.")
    st.markdown(f"This section guides Maya through loading her new credit application dataset and configuring parameters for a tailored assessment. Defining these parameters upfront is crucial to align the assessment with Software Innovators Inc.'s specific model requirements and compliance policies.")

    st.subheader("Upload Datasets")
    primary_uploaded_file = st.file_uploader("Upload Primary Dataset (CSV)", type=["csv"], key="primary_uploader")
    baseline_uploaded_file = st.file_uploader("Upload Optional Baseline Dataset (CSV) for Drift Detection", type=["csv"], key="baseline_uploader")

    if primary_uploaded_file:
        st.session_state['primary_data_filename'] = primary_uploaded_file.name
        st.session_state['primary_temp_path'] = _save_uploaded_file(primary_uploaded_file)
        st.session_state['primary_df'] = pd.read_csv(st.session_state['primary_temp_path'])
        st.markdown(f"**Primary Dataset Preview:**")
        st.dataframe(st.session_state['primary_df'].head())
    else:
        st.session_state['primary_df'] = None
        st.session_state['primary_data_filename'] = None
        st.session_state['primary_temp_path'] = None

    if baseline_uploaded_file:
        st.session_state['baseline_data_filename'] = baseline_uploaded_file.name
        st.session_state['baseline_temp_path'] = _save_uploaded_file(baseline_uploaded_file)
        st.session_state['baseline_df'] = pd.read_csv(st.session_state['baseline_temp_path'])
        st.markdown(f"**Baseline Dataset Preview:**")
        st.dataframe(st.session_state['baseline_df'].head())
    else:
        st.session_state['baseline_df'] = None
        st.session_state['baseline_data_filename'] = None
        st.session_state['baseline_temp_path'] = None

    if st.session_state['primary_df'] is not None:
        all_columns = st.session_state['primary_df'].columns.tolist()

        st.subheader("Configure Assessment Parameters")
        target_col_input = st.selectbox("Select Target Label Column", all_columns, key="target_col_input")
        sensitive_cols_input = st.multiselect("Select Sensitive Attributes (for Bias Detection)", all_columns, key="sensitive_cols_input")

        st.markdown(f"For Software Innovators Inc., a key concern is ensuring fair lending practices. Therefore, identifying sensitive attributes like `{', '.join(sensitive_cols_input) if sensitive_cols_input else '...'}` is paramount to later check for potential biases.")

        # Protected Groups configuration (default example provided for credit data)
        with st.expander("Define Protected Groups (e.g., Privileged/Unprivileged)"):
            st.markdown(f"Specify the privileged and unprivileged groups for each sensitive attribute. These definitions are crucial for bias metric computation.")
            configured_protected_groups = {}
            for attr in sensitive_cols_input:
                st.markdown(f"**Sensitive Attribute: `{attr}`**")
                privileged_default = "Married" if attr == "marital_status" else "North" if attr == "region" else ""
                unprivileged_default = "Single,Divorced" if attr == "marital_status" else "South,East,West" if attr == "region" else ""
                
                privileged_group = st.text_input(f"Privileged group for `{attr}`:", value=privileged_default, key=f"privileged_{attr}")
                unprivileged_groups_str = st.text_input(f"Unprivileged groups for `{attr}` (comma-separated):", value=unprivileged_default, key=f"unprivileged_{attr}")
                
                if privileged_group and unprivileged_groups_str:
                    configured_protected_groups[attr] = {
                        'privileged': privileged_group,
                        'unprivileged': [g.strip() for g in unprivileged_groups_str.split(',')]
                    }
                else:
                    st.warning(f"Please define both privileged and unprivileged groups for `{attr}` if you wish to include it in bias analysis.")
            
            # Example default:
            if not sensitive_cols_input:
                st.info("No sensitive attributes selected. Protected groups configuration is optional.")
            elif not configured_protected_groups:
                st.warning("No protected groups configured. Bias metrics may not be computed correctly.")


        # Threshold Overrides
        custom_thresholds = {}
        with st.expander("Override Default Thresholds"):
            st.markdown(f"Maya can override predefined warning and failure thresholds for data quality, bias, and drift metrics to align with Software Innovators Inc.'s specific risk appetite.")
            st.markdown(f"### Data Quality Thresholds")
            st.markdown(f"**Missingness Ratio:** (> value WARN, > value FAIL)")
            dq_missing_warn = st.number_input("Missingness WARN Ratio", value=0.05, format="%.2f", key="dq_missing_warn")
            dq_missing_fail = st.number_input("Missingness FAIL Ratio", value=0.20, format="%.2f", key="dq_missing_fail")
            custom_thresholds['data_quality_thresholds'] = {'missingness_ratio': {'warn': dq_missing_warn, 'fail': dq_missing_fail}}
            # ... (Add other DQ thresholds like duplicates, type_inconsistency, range_violation, cardinality)
            # For brevity, only missingness shown here, but other data quality thresholds from source.py should be included.
            st.markdown(f"**Duplicate Rows Ratio:** (> value WARN, > value FAIL)")
            dq_dup_warn = st.number_input("Duplicate Rows WARN Ratio", value=0.01, format="%.2f", key="dq_dup_warn")
            dq_dup_fail = st.number_input("Duplicate Rows FAIL Ratio", value=0.05, format="%.2f", key="dq_dup_fail")
            custom_thresholds['data_quality_thresholds']['duplicate_rows_ratio'] = {'warn': dq_dup_warn, 'fail': dq_dup_fail}

            st.markdown(f"**Type Inconsistency Ratio:** (> value WARN, > value FAIL)")
            dq_type_warn = st.number_input("Type Inconsistency WARN Ratio", value=0.00, format="%.2f", key="dq_type_warn")
            dq_type_fail = st.number_input("Type Inconsistency FAIL Ratio", value=0.00, format="%.2f", key="dq_type_fail")
            custom_thresholds['data_quality_thresholds']['type_inconsistency_ratio'] = {'warn': dq_type_warn, 'fail': dq_type_fail}

            st.markdown(f"**Range Violation Ratio:** (> value WARN, > value FAIL)")
            dq_range_warn = st.number_input("Range Violation WARN Ratio", value=0.00, format="%.2f", key="dq_range_warn")
            dq_range_fail = st.number_input("Range Violation FAIL Ratio", value=0.00, format="%.2f", key="dq_range_fail")
            custom_thresholds['data_quality_thresholds']['range_violation_ratio'] = {'warn': dq_range_warn, 'fail': dq_range_fail}

            st.markdown(f"**Cardinality Unique Count (Min):** (< value WARN, < value FAIL)")
            dq_card_min_warn = st.number_input("Cardinality Min WARN Count", value=2, key="dq_card_min_warn")
            dq_card_min_fail = st.number_input("Cardinality Min FAIL Count", value=1, key="dq_card_min_fail")
            custom_thresholds['data_quality_thresholds']['cardinality_unique_count_min'] = {'warn': dq_card_min_warn, 'fail': dq_card_min_fail}

            st.markdown(f"**Cardinality Unique Count (Max Ratio):** (> value WARN, > value FAIL)")
            dq_card_max_warn = st.number_input("Cardinality Max WARN Ratio (e.g., 0.5 for >50% unique)", value=0.5, format="%.2f", key="dq_card_max_warn")
            dq_card_max_fail = st.number_input("Cardinality Max FAIL Ratio (e.g., 0.9 for >90% unique)", value=0.9, format="%.2f", key="dq_card_max_fail")
            custom_thresholds['data_quality_thresholds']['cardinality_unique_count_max_ratio'] = {'warn': dq_card_max_warn, 'fail': dq_card_max_fail}

            # Feature Range Expectations (example for Age, Income, Credit Score, Loan Amount)
            st.markdown(f"### Feature Range Expectations (Numerical Features)")
            configured_feature_ranges = {}
            numeric_cols = st.session_state['primary_df'].select_dtypes(include=np.number).columns.tolist()
            for col in numeric_cols:
                col_range_expander = st.expander(f"Set Range for '{col}'")
                with col_range_expander:
                    min_val = st.number_input(f"Min value for '{col}'", value=0.0 if 'income' in col or 'loan_amount' in col else 18.0 if 'age' in col else 300.0 if 'credit_score' in col else 0.0, key=f"range_min_{col}")
                    max_val = st.number_input(f"Max value for '{col}'", value=500000.0 if 'income' in col else 100000.0 if 'loan_amount' in col else 90.0 if 'age' in col else 850.0 if 'credit_score' in col else 1000.0, key=f"range_max_{col}")
                    configured_feature_ranges[col] = {'min': min_val, 'max': max_val}
            custom_thresholds['feature_range_expectations'] = configured_feature_ranges


            st.markdown(f"### Bias Thresholds")
            st.markdown(f"**Demographic Parity Difference:** (Abs > value WARN, Abs > value FAIL)")
            bias_dpd_warn = st.number_input("DPD WARN Threshold", value=0.10, format="%.2f", key="bias_dpd_warn")
            bias_dpd_fail = st.number_input("DPD FAIL Threshold", value=0.20, format="%.2f", key="bias_dpd_fail")
            custom_thresholds['bias_thresholds'] = {'demographic_parity_difference': {'warn': bias_dpd_warn, 'fail': bias_dpd_fail}}

            st.markdown(f"**Disparate Impact Ratio:** (< lower_value or > upper_value WARN, < lower_value or > upper_value FAIL)")
            bias_dir_warn_l = st.number_input("DIR WARN Lower Bound", value=0.80, format="%.2f", key="bias_dir_warn_l")
            bias_dir_warn_u = st.number_input("DIR WARN Upper Bound", value=1.25, format="%.2f", key="bias_dir_warn_u")
            bias_dir_fail_l = st.number_input("DIR FAIL Lower Bound", value=0.67, format="%.2f", key="bias_dir_fail_l")
            bias_dir_fail_u = st.number_input("DIR FAIL Upper Bound", value=1.50, format="%.2f", key="bias_dir_fail_u")
            custom_thresholds['bias_thresholds']['disparate_impact_ratio'] = {'warn_lower': bias_dir_warn_l, 'warn_upper': bias_dir_warn_u, 'fail_lower': bias_dir_fail_l, 'fail_upper': bias_dir_fail_u}

            st.markdown(f"**Proxy TPR Gap:** (Abs > value WARN, Abs > value FAIL)")
            bias_tpr_warn = st.number_input("TPR Gap WARN Threshold", value=0.10, format="%.2f", key="bias_tpr_warn")
            bias_tpr_fail = st.number_input("TPR Gap FAIL Threshold", value=0.20, format="%.2f", key="bias_tpr_fail")
            custom_thresholds['bias_thresholds']['proxy_tpr_gap'] = {'warn': bias_tpr_warn, 'fail': bias_tpr_fail}
            
            st.markdown(f"**Proxy FPR Gap:** (Abs > value WARN, Abs > value FAIL)")
            bias_fpr_warn = st.number_input("FPR Gap WARN Threshold", value=0.10, format="%.2f", key="bias_fpr_warn")
            bias_fpr_fail = st.number_input("FPR Gap FAIL Threshold", value=0.20, format="%.2f", key="bias_fpr_fail")
            custom_thresholds['bias_thresholds']['proxy_fpr_gap'] = {'warn': bias_fpr_warn, 'fail': bias_fpr_fail}


            st.markdown(f"### Drift Thresholds (PSI)")
            st.markdown(f"**Population Stability Index (PSI):** (> value WARN, > value FAIL)")
            drift_psi_warn = st.number_input("PSI WARN Threshold", value=0.10, format="%.2f", key="drift_psi_warn")
            drift_psi_fail = st.number_input("PSI FAIL Threshold", value=0.25, format="%.2f", key="drift_psi_fail")
            custom_thresholds['drift_thresholds'] = {'psi': {'warn': drift_psi_warn, 'fail': drift_psi_fail}}


        if st.button("Apply Configuration", key="apply_config_btn"):
            if st.session_state['primary_df'] is not None and target_col_input and sensitive_cols_input:
                try:
                    # The source.py function expects paths, so we need to save temp files.
                    # We have already saved them above after upload.
                    
                    # Merge configured_protected_groups into custom_thresholds for convenience if needed,
                    # or pass separately as per load_and_configure_datasets signature.
                    # As per source.py, load_and_configure_datasets takes sensitive_attributes and config_thresholds.
                    # It has its own default protected_groups, which we would override if passed in config_thresholds.
                    # Let's align by ensuring the custom_thresholds includes protected_groups.
                    custom_thresholds['protected_groups'] = configured_protected_groups

                    st.session_state['primary_df'], st.session_state['baseline_df'], st.session_state['assessment_config'] = \
                        load_and_configure_datasets(
                            st.session_state['primary_temp_path'],
                            target_col_input,
                            sensitive_cols_input,
                            baseline_path=st.session_state['baseline_temp_path'],
                            config_thresholds=custom_thresholds
                        )
                    st.success("Configuration Applied Successfully!")
                    st.session_state['config_applied'] = True
                    # Reset analysis results to force re-computation with new config
                    st.session_state['data_quality_results'] = None
                    st.session_state['bias_metrics_results'] = None
                    st.session_state['drift_detection_results'] = {'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
                    st.session_state['readiness_decision'] = 'Pending Configuration'
                except Exception as e:
                    st.error(f"Error applying configuration: {e}")
                    st.session_state['config_applied'] = False
            else:
                st.warning("Please upload a primary dataset, select a target column, and sensitive attributes to apply configuration.")
        
        if st.session_state['assessment_config']:
            st.subheader("Current Assessment Configuration:")
            st.json(st.session_state['assessment_config'])
            st.markdown(f"Maya has successfully loaded the datasets and reviewed the configuration. She can see the target column, the sensitive attributes, and the default (and any custom) thresholds for each metric. This initial setup is critical to ensure the assessment is aligned with project goals and regulatory requirements.")
    else:
        st.info("Please upload a primary dataset to proceed with configuration.")
```

#### Page: Data Quality

```python
if st.session_state['current_page'] == "Data Quality":
    st.markdown(f"# 2. Core Data Quality Assessment: Uncovering Raw Data Issues")
    st.markdown(f"Before any sophisticated modeling, Maya must ensure the fundamental quality of the dataset. This means checking for common issues like missing values, duplicate entries, inconsistent data types, values outside expected ranges, and inappropriate cardinality for categorical features. Catching these problems early prevents downstream errors in model training, improves model robustness, and saves significant computational resources. For Software Innovators Inc., poor data quality could lead to inaccurate credit risk assessments, violating internal policies and potentially regulatory guidelines.")
    st.markdown(f"Maya will use the configured thresholds to assign a 'PASS', 'WARN', or 'FAIL' status to each quality aspect of every feature.")

    st.markdown(r"**Missingness Ratio ($M_i$):** The proportion of missing values for feature $i$.")
    st.markdown(r"$$ M_i = \frac{{\text{{Number of Missing Values in Feature }} i}}{{\text{{Total Number of Rows}}}} $$")
    st.markdown(r"where $M_i$ is the missingness ratio for feature $i$, Number of Missing Values in Feature $i$ is the count of null values in column $i$, and Total Number of Rows is the total number of entries in the dataset.")

    st.markdown(r"**Duplicate Rows Ratio ($D$):** The proportion of rows that are exact duplicates of other rows in the dataset.")
    st.markdown(r"$$ D = \frac{{\text{{Number of Duplicate Rows}}}}{{\text{{Total Number of Rows}}}} $$")
    st.markdown(r"where $D$ is the duplicate rows ratio, Number of Duplicate Rows is the count of rows that are identical to another row, and Total Number of Rows is the total number of entries in the dataset.")

    st.markdown(f"**Type Inconsistency:** Measured as the ratio of non-conforming data types within a column. For example, if a numeric column contains string values, this metric will be high.")

    st.markdown(r"**Range Violation Ratio:** For numerical features, this is the ratio of values falling outside a predefined acceptable range.")
    st.markdown(r"$$ R_i = \frac{{\text{{Number of Values Outside Expected Range for Feature }} i}}{{\text{{Total Number of Rows}}}} $$")
    st.markdown(r"where $R_i$ is the range violation ratio for feature $i$, Number of Values Outside Expected Range for Feature $i$ is the count of values in column $i$ that are outside the specified min/max range, and Total Number of Rows is the total number of entries in the dataset.")

    st.markdown(f"**Cardinality Check:** For categorical features, this examines the number of unique values. Extremely low cardinality (e.g., only one unique value) or extremely high cardinality (e.g., unique values approaching the total number of rows) can indicate issues.")

    if not st.session_state.get('config_applied'):
        st.warning("Please go to 'Data Upload & Configuration' to upload data and apply configuration first.")
    elif st.session_state['primary_df'] is None:
        st.warning("Primary dataset not loaded. Please upload data in 'Data Upload & Configuration' page.")
    else:
        if st.button("Perform Data Quality Checks", key="run_dq_checks"):
            with st.spinner("Running Data Quality Checks..."):
                st.session_state['data_quality_results'] = perform_data_quality_checks(st.session_state['primary_df'], st.session_state['assessment_config'])
            st.success("Data Quality Checks Completed!")

        if st.session_state['data_quality_results']:
            st.subheader("Data Quality Assessment Results")
            overall_dq_status = st.session_state['data_quality_results']['overall_dataset_quality_status']
            st.markdown(f"**Overall Data Quality Status: {overall_dq_status}**")
            
            # Prepare data for display
            table_data = []
            dq_thresholds = st.session_state['assessment_config']['data_quality_thresholds']

            # Dataset overall duplicates
            dup_res = st.session_state['data_quality_results']['dataset_overall']
            table_data.append({
                "Feature": "Dataset (Overall)",
                "Metric": "Duplicate Rows Ratio",
                "Value": f"{dup_res['duplicate_rows_ratio']:.2%}",
                "Status": dup_res['duplicate_rows_status'],
                "Threshold (Warn/Fail)": f">{dq_thresholds['duplicate_rows_ratio']['warn']:.2%} (W) / >{dq_thresholds['duplicate_rows_ratio']['fail']:.2%} (F)"
            })

            for col, metrics in st.session_state['data_quality_results'].items():
                if col in ['dataset_overall', 'overall_dataset_quality_status']:
                    continue

                # Missingness
                table_data.append({
                    "Feature": col,
                    "Metric": "Missingness Ratio",
                    "Value": f"{metrics['missing_ratio']:.2%}",
                    "Status": metrics['missing_status'],
                    "Threshold (Warn/Fail)": f">{dq_thresholds['missingness_ratio']['warn']:.2%} (W) / >{dq_thresholds['missingness_ratio']['fail']:.2%} (F)"
                })
                
                # Type Consistency
                table_data.append({
                    "Feature": col,
                    "Metric": "Type Inconsistency Ratio",
                    "Value": f"{metrics['type_inconsistency_ratio']:.2%} (Consistent Type: {metrics['consistent_type']})",
                    "Status": metrics['type_consistency_status'],
                    "Threshold (Warn/Fail)": f">{dq_thresholds['type_inconsistency_ratio']['warn']:.2%} (W) / >{dq_thresholds['type_inconsistency_ratio']['fail']:.2%} (F)"
                })

                # Range Violations
                if metrics['range_violation_status'] != 'N/A':
                    table_data.append({
                        "Feature": col,
                        "Metric": "Range Violation Ratio",
                        "Value": f"{metrics['range_violation_ratio']:.2%} (Expected: {metrics.get('range_min_expected', 'N/A')}-{metrics.get('range_max_expected', 'N/A')})",
                        "Status": metrics['range_violation_status'],
                        "Threshold (Warn/Fail)": f">{dq_thresholds['range_violation_ratio']['warn']:.2%} (W) / >{dq_thresholds['range_violation_ratio']['fail']:.2%} (F)"
                    })
                
                # Cardinality
                table_data.append({
                    "Feature": col,
                    "Metric": "Unique Values Count",
                    "Value": f"{metrics['unique_values_count']}",
                    "Status": metrics['cardinality_status'],
                    "Threshold (Warn/Fail)": f"< {dq_thresholds['cardinality_unique_count_min']['warn']} or > {dq_thresholds['cardinality_unique_count_max_ratio']['warn']:.0%} (W) / < {dq_thresholds['cardinality_unique_count_min']['fail']} or > {dq_thresholds['cardinality_unique_count_max_ratio']['fail']:.0%} (F)"
                })
            
            # Display the results in a Streamlit DataFrame
            st.dataframe(pd.DataFrame(table_data), height=400) # Increased height to show more rows

            st.markdown(f"---")
            st.markdown(f"### Explanation of Execution")
            st.markdown(f"Maya's data quality assessment reveals critical insights. The table clearly shows features flagged with 'WARN' or 'FAIL'. For example, `income` might have a 'FAIL' for missingness, indicating a significant portion of income data is absent. `credit_score` might show a 'FAIL' for type inconsistency due to non-numeric entries, which would break any numerical operations. `age` could have a 'WARN' for range violations, requiring a cleanup of outlier values.")
            st.markdown(f"These findings directly inform Maya's next steps:")
            st.markdown(f"-   **Missing Data:** Maya must decide on an imputation strategy (e.g., mean, median, predictive imputation) or whether the column should be dropped.")
            st.markdown(f"-   **Type Inconsistencies:** Requires data cleaning to convert values to the correct type or remove corrupted entries.")
            st.markdown(f"-   **Range Violations:** Outliers need to be handled, either corrected, capped, or removed, to prevent skewed model learning.")
            st.markdown(f"By addressing these issues proactively, Maya ensures the credit approval model receives reliable data, preventing it from making decisions based on faulty inputs.")

```

#### Page: Bias Metrics

```python
if st.session_state['current_page'] == "Bias Metrics":
    st.markdown(f"# 3. Bias Metric Computation: Ensuring Fairness in Data")
    st.markdown(f"At Software Innovators Inc., ensuring fairness and avoiding discrimination in credit decisions is not just a regulatory requirement but a core ethical principle. Maya understands that biases present in the training data can be learned and amplified by models, leading to unfair outcomes for certain demographic groups. Before training the credit approval model, she must quantify any inherent biases within the new dataset. This assessment helps her identify if the raw data itself exhibits disparities in credit repayment outcomes across sensitive attributes like `marital_status` or `region`.")
    st.markdown(f"Since we are in a pre-training context (no model predictions yet), we will focus on measuring statistical parity and outcome disparities based on the *actual* target variable distributions across protected groups.")

    st.markdown(r"**Demographic Parity Difference (DPD):** Measures the difference in the proportion of the favorable outcome (e.g., loan repaid) between an unprivileged group and a privileged group. A value close to 0 indicates demographic parity.")
    st.markdown(r"$$ DPD = P(Y=1 | A_{\text{unprivileged}}) - P(Y=1 | A_{\text{privileged}}) $$")
    st.markdown(r"where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.")

    st.markdown(r"**Disparate Impact Ratio (DIR):** Measures the ratio of the favorable outcome proportion for the unprivileged group to the privileged group. A value near 1 suggests no disparate impact. Values significantly below 1 (e.g., < 0.8) indicate the unprivileged group is less likely to receive the favorable outcome, while values significantly above 1 (e.g., > 1.25) indicate the unprivileged group is more likely.")
    st.markdown(r"$$ DIR = \frac{{P(Y=1 | A_{\text{unprivileged}})}}{{P(Y=1 | A_{\text{privileged}})}} $$")
    st.markdown(r"where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.")

    st.markdown(r"**Proxy True Positive Rate Gap (TPR Gap):** In a pre-training context, this can be interpreted as the difference in the *actual positive outcome rate* (prevalence of $Y=1$) between unprivileged and privileged groups. A value close to 0 indicates similar rates of positive outcomes for both groups in the raw data.")
    st.markdown(r"$$ \text{{Proxy TPR Gap}} = P(Y=1 | A_{\text{unprivileged}}) - P(Y=1 | A_{\text{privileged}}) $$")
    st.markdown(r"where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.")

    st.markdown(r"**Proxy False Positive Rate Gap (FPR Gap):** Similarly, in a pre-training context, this can be interpreted as the difference in the *actual negative outcome rate* (prevalence of $Y=0$) between unprivileged and privileged groups. A value close to 0 indicates similar rates of negative outcomes for both groups in the raw data.")
    st.markdown(r"$$ \text{{Proxy FPR Gap}} = P(Y=0 | A_{\text{unprivileged}}) - P(Y=0 | A_{\text{privileged}}) $$")
    st.markdown(r"where $Y=0$ is the unfavorable outcome (e.g., loan not repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.")

    if not st.session_state.get('config_applied'):
        st.warning("Please go to 'Data Upload & Configuration' to upload data and apply configuration first.")
    elif st.session_state['primary_df'] is None:
        st.warning("Primary dataset not loaded. Please upload data in 'Data Upload & Configuration' page.")
    elif not st.session_state['assessment_config'].get('sensitive_attributes'):
        st.info("No sensitive attributes configured. Bias metrics cannot be computed.")
    else:
        if st.button("Compute Bias Metrics", key="run_bias_checks"):
            with st.spinner("Computing Bias Metrics..."):
                try:
                    st.session_state['bias_metrics_results'] = compute_bias_metrics(
                        st.session_state['primary_df'],
                        st.session_state['assessment_config']['target_column'],
                        st.session_state['assessment_config']['sensitive_attributes'],
                        st.session_state['assessment_config']['protected_groups'],
                        st.session_state['assessment_config']
                    )
                    st.success("Bias Metrics Computed!")
                except ValueError as e:
                    st.error(f"Error computing bias metrics: {e}")
                    st.session_state['bias_metrics_results'] = None

        if st.session_state['bias_metrics_results']:
            st.subheader("Bias Metrics Assessment Results")
            overall_bias_status = st.session_state['bias_metrics_results']['overall_bias_status']
            st.markdown(f"**Overall Bias Status: {overall_bias_status}**")

            table_data = []
            bias_thresholds = st.session_state['assessment_config']['bias_thresholds']

            for attr, metrics in st.session_state['bias_metrics_results'].items():
                if attr == 'overall_bias_status':
                    continue
                
                # Demographic Parity Difference
                warn_dpd = bias_thresholds['demographic_parity_difference']['warn']
                fail_dpd = bias_thresholds['demographic_parity_difference']['fail']
                table_data.append({
                    "Sensitive Attribute": attr,
                    "Metric": "Demographic Parity Difference",
                    "Value": f"{metrics['demographic_parity_difference']:.4f}",
                    "Status": metrics['demographic_parity_difference_status'],
                    "Threshold (Warn/Fail)": f"Abs > {warn_dpd:.2f} (W) / > {fail_dpd:.2f} (F)"
                })

                # Disparate Impact Ratio
                warn_dir_l = bias_thresholds['disparate_impact_ratio']['warn_lower']
                warn_dir_u = bias_thresholds['disparate_impact_ratio']['warn_upper']
                fail_dir_l = bias_thresholds['disparate_impact_ratio']['fail_lower']
                fail_dir_u = bias_thresholds['disparate_impact_ratio']['fail_upper']
                table_data.append({
                    "Sensitive Attribute": attr,
                    "Metric": "Disparate Impact Ratio",
                    "Value": f"{metrics['disparate_impact_ratio']:.4f}",
                    "Status": metrics['disparate_impact_ratio_status'],
                    "Threshold (Warn/Fail)": f"< {warn_dir_l:.2f} or > {warn_dir_u:.2f} (W) / < {fail_dir_l:.2f} or > {fail_dir_u:.2f} (F)"
                })

                # Proxy TPR Gap
                warn_tpr_gap = bias_thresholds['proxy_tpr_gap']['warn']
                fail_tpr_gap = bias_thresholds['proxy_tpr_gap']['fail']
                table_data.append({
                    "Sensitive Attribute": attr,
                    "Metric": "Proxy TPR Gap",
                    "Value": f"{metrics['proxy_tpr_gap']:.4f}",
                    "Status": metrics['proxy_tpr_gap_status'],
                    "Threshold (Warn/Fail)": f"Abs > {warn_tpr_gap:.2f} (W) / > {fail_tpr_gap:.2f} (F)"
                })

                # Proxy FPR Gap
                warn_fpr_gap = bias_thresholds['proxy_fpr_gap']['warn']
                fail_fpr_gap = bias_thresholds['proxy_fpr_gap']['fail']
                table_data.append({
                    "Sensitive Attribute": attr,
                    "Metric": "Proxy FPR Gap",
                    "Value": f"{metrics['proxy_fpr_gap']:.4f}",
                    "Status": metrics['proxy_fpr_gap_status'],
                    "Threshold (Warn/Fail)": f"Abs > {warn_fpr_gap:.2f} (W) / > {fail_fpr_gap:.2f} (F)"
                })
            
            st.dataframe(pd.DataFrame(table_data), height=400)

            st.markdown(f"---")
            st.markdown(f"### Explanation of Execution")
            st.markdown(f"The bias metrics report provides Maya with quantitative evidence of fairness (or lack thereof) in the raw data. For instance, if the `region` attribute shows a Disparate Impact Ratio significantly below 1 (e.g., for the 'South' region), it indicates that customers from this region are less likely to have favorable credit outcomes ($Y=1$) in the dataset compared to the privileged 'North' region. This is a critical 'WARN' or 'FAIL' condition for Software Innovators Inc.")
            st.markdown(f"Maya must now consider strategies to mitigate these biases *before* model training. This could involve:")
            st.markdown(f"-   **Data collection review:** Investigating if the data collection process itself introduced biases.")
            st.markdown(f"-   **Feature engineering:** Creating new features that might reduce reliance on sensitive attributes.")
            st.markdown(f"-   **Resampling techniques:** Over-sampling underrepresented groups or under-sampling overrepresented groups to balance the dataset.")
            st.markdown(f"These steps are crucial for Software Innovators Inc. to ensure equitable credit access and avoid legal or reputational risks.")
```

#### Page: Drift

```python
if st.session_state['current_page'] == "Drift":
    st.markdown(f"# 4. Drift Detection: Monitoring Data Distribution Shifts")
    st.markdown(f"Software Innovators Inc. operates in a dynamic financial market, where customer behaviors and economic conditions can change rapidly. Maya understands that a credit approval model trained on old data might become less accurate if the underlying data distribution shifts over time. This phenomenon, known as \"data drift,\" can severely degrade model performance in production. To mitigate this risk, Maya needs to compare the new credit application dataset against a historical \"baseline\" dataset (the data the current production model was originally trained on).")

    st.markdown(r"The **Population Stability Index (PSI)** is a widely used metric to quantify data drift for numerical features. It measures how much a variable's distribution has changed from a baseline period to a current period.")
    st.markdown(r"**Population Stability Index (PSI):** For each feature, the PSI is calculated by:")
    st.markdown(r"1.  Dividing the range of the feature into several bins (e.g., 10 bins).")
    st.markdown(r"2.  Calculating the percentage of observations in each bin for both the baseline ($P_{{0i}}$) and the current ($P_i$) datasets.")
    st.markdown(r"3.  Summing the contributions from each bin:")
    st.markdown(r"$$ \text{{PSI}} = \sum_{{i=1}}^{{N}} (P_i - P_{{0i}}) \times \ln\left(\frac{{P_i}}{{P_{{0i}}}}\right) $$")
    st.markdown(r"where $N$ is the number of bins, $P_i$ is the percentage of observations in bin $i$ for the current dataset, and $P_{{0i}}$ is the percentage of observations in bin $i$ for the baseline dataset. A small constant (epsilon) is typically added to $P_i$ and $P_{{0i}}$ to avoid issues with zero values in the logarithm.")
    st.markdown(f"Common interpretations for PSI:")
    st.markdown(f"-   PSI < 0.1: No significant shift (PASS)")
    st.markdown(f"-   0.1 <= PSI < 0.25: Small shift (WARN)")
    st.markdown(f"-   PSI >= 0.25: Significant shift (FAIL)")

    if not st.session_state.get('config_applied'):
        st.warning("Please go to 'Data Upload & Configuration' to upload data and apply configuration first.")
    elif st.session_state['primary_df'] is None:
        st.warning("Primary dataset not loaded. Please upload data in 'Data Upload & Configuration' page.")
    elif st.session_state['baseline_df'] is None:
        st.info("Baseline dataset not provided. Drift detection will be skipped.")
        st.session_state['drift_detection_results'] = {'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
    else:
        if st.button("Detect Data Drift (PSI)", key="run_drift_checks"):
            with st.spinner("Detecting Data Drift..."):
                numeric_columns = st.session_state['primary_df'].select_dtypes(include=np.number).columns.tolist()
                target_col = st.session_state['assessment_config']['target_column']
                if target_col in numeric_columns:
                    numeric_columns.remove(target_col)

                try:
                    st.session_state['drift_detection_results'] = calculate_psi(
                        st.session_state['primary_df'],
                        st.session_state['baseline_df'],
                        numeric_columns,
                        st.session_state['assessment_config']
                    )
                    st.success("Drift Detection Completed!")
                except Exception as e:
                    st.error(f"Error calculating PSI: {e}")
                    st.session_state['drift_detection_results'] = {'overall_drift_status': 'ERROR', 'message': f'Error calculating PSI: {e}'}

        if st.session_state['drift_detection_results'] and st.session_state['drift_detection_results']['overall_drift_status'] != 'N/A':
            st.subheader("Drift Detection Results (Population Stability Index - PSI)")
            overall_drift_status = st.session_state['drift_detection_results']['overall_drift_status']
            st.markdown(f"**Overall Drift Status: {overall_drift_status}**")

            table_data = []
            psi_thresholds = st.session_state['assessment_config']['drift_thresholds']['psi']

            for col, metrics in st.session_state['drift_detection_results'].items():
                if col == 'overall_drift_status' or col == 'message':
                    continue
                
                warn_psi = psi_thresholds['warn']
                fail_psi = psi_thresholds['fail']
                psi_val_str = f"{metrics['psi']:.4f}" if not np.isnan(metrics['psi']) else metrics.get('error', 'N/A')
                
                table_data.append({
                    "Feature": col,
                    "PSI Value": psi_val_str,
                    "Status": metrics['status'],
                    "Threshold (Warn/Fail)": f"> {warn_psi:.2f} (W) / > {fail_psi:.2f} (F)"
                })
            
            st.dataframe(pd.DataFrame(table_data), height=400)

            st.subheader("Visualizing Drift for Key Numerical Features")
            # Select relevant features for visualization
            features_to_plot = [
                col for col in ['age', 'income', 'credit_score', 'loan_amount']
                if col in st.session_state['primary_df'].columns and col in st.session_state['baseline_df'].columns and col in st.session_state['drift_detection_results']
            ]
            
            if features_to_plot:
                num_plots = len(features_to_plot)
                num_cols = 2
                num_rows = (num_plots + num_cols - 1) // num_cols # Ceiling division
                
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
                axes = axes.flatten() if num_plots > 1 else [axes] # Ensure axes is iterable

                for i, col in enumerate(features_to_plot):
                    ax = axes[i]
                    psi_val = st.session_state['drift_detection_results'][col].get('psi', np.nan)
                    
                    sns.histplot(st.session_state['primary_df'][col].dropna(), kde=True, color='skyblue', label='Primary', ax=ax, stat='density', alpha=0.7)
                    sns.histplot(st.session_state['baseline_df'][col].dropna(), kde=True, color='salmon', label='Baseline', ax=ax, stat='density', alpha=0.7)
                    ax.set_title(f'{col} Distribution (PSI: {psi_val:.2f})')
                    ax.legend()
                
                # Hide unused subplots if num_plots is odd and num_cols is 2
                for j in range(num_plots, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No common numerical features found or selected for drift visualization.")

            st.markdown(f"---")
            st.markdown(f"### Explanation of Execution")
            st.markdown(f"Maya examines the PSI results to detect significant shifts. For instance, if `income` or `credit_score` show a 'WARN' or 'FAIL' PSI value, it means the distribution of these critical features in the new dataset has changed notably from the baseline. The visualizations provide an intuitive understanding of these shifts. A shift in `income` distribution, for example, could indicate changing economic conditions or a different customer segment applying for credit.")
            st.markdown(f"This insight is crucial for Maya:")
            st.markdown(f"-   **Model Re-training:** A significant drift (`FAIL`) in key features might necessitate immediate re-training of the credit approval model using the new data.")
            st.markdown(f"-   **Monitoring Strategy:** Even a 'WARN' suggests closer monitoring of the model's performance on these features in production.")
            st.markdown(f"-   **Business Context:** It prompts Software Innovators Inc. to investigate the business reasons behind the drift, informing both model development and business strategy.")
            st.markdown(f"By systematically identifying data drift, Maya helps Software Innovators Inc. maintain accurate and relevant models, preventing unexpected performance degradation in live environments.")
        else:
            st.info(st.session_state['drift_detection_results']['message'])

```

#### Page: Summary & Export

```python
if st.session_state['current_page'] == "Summary & Export":
    st.markdown(f"# 5. Overall Readiness Decision and Comprehensive Reporting")
    st.markdown(f"After conducting thorough data quality, bias, and drift assessments, Maya needs to synthesize all findings into a clear, deterministic decision regarding the dataset's readiness for model training. This final decision, along with a comprehensive report, is critical for communication with Model Validators (Persona 2) and Risk/Compliance Partners (Persona 3) at Software Innovators Inc. It provides a transparent, quantitative basis for proceeding or halting model development, ensuring accountability and adherence to governance standards.")
    st.markdown(f"The overall readiness decision follows a strict logic:")
    st.markdown(f"-   **DO NOT DEPLOY:** If any 'FAIL' condition is identified across data quality, bias, or drift metrics. This indicates severe issues that must be addressed before proceeding.")
    st.markdown(f"-   **PROCEED WITH MITIGATION:** If there are only 'WARN' conditions, but no 'FAIL's. This means the dataset has minor issues that can likely be mitigated (e.g., specific data cleaning, bias-aware modeling) while proceeding with development.")
    st.markdown(f"-   **PROCEED:** If all checks result in a 'PASS'. This indicates the dataset is robust and ready for model training without immediate, significant concerns.")

    if not st.session_state.get('config_applied') or st.session_state['data_quality_results'] is None or st.session_state['bias_metrics_results'] is None:
        st.warning("Please complete the 'Data Upload & Configuration', 'Data Quality', and 'Bias Metrics' steps first.")
    else:
        # Ensure drift results are initialized, even if no baseline was provided
        if st.session_state['drift_detection_results'] is None:
             st.session_state['drift_detection_results'] = {'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
             
        if st.button("Generate Final Report and Export Artifacts", key="generate_reports_btn"):
            with st.spinner("Generating reports and bundling artifacts..."):
                st.session_state['readiness_decision'] = make_readiness_decision(
                    st.session_state['data_quality_results'],
                    st.session_state['bias_metrics_results'],
                    st.session_state['drift_detection_results']
                )

                # Ensure temp_data directory exists for report generation if not already created.
                os.makedirs("data", exist_ok=True) # Source.py generates sample data in 'data' directory. Reports also need 'reports'
                os.makedirs("reports", exist_ok=True)

                st.session_state['reports_folder_path'] = generate_reports(
                    st.session_state['data_quality_results'],
                    st.session_state['bias_metrics_results'],
                    st.session_state['drift_detection_results'],
                    st.session_state['readiness_decision'],
                    st.session_state['assessment_config'],
                    st.session_state['primary_temp_path'] if st.session_state['primary_temp_path'] else st.session_state['primary_data_filename'], # Pass original filename or temp path
                    st.session_state['baseline_temp_path'] if st.session_state['baseline_temp_path'] else st.session_state['baseline_data_filename']
                )

                run_id = os.path.basename(st.session_state['reports_folder_path']).replace('run_', '')
                st.session_state['zip_archive_path'] = bundle_artifacts(st.session_state['reports_folder_path'], run_id)
            
            st.success("Reports Generated and Artifacts Bundled!")

        if st.session_state['readiness_decision'] != 'Pending Configuration':
            st.subheader("Overall Dataset Readiness Decision:")
            st.markdown(f"## **{st.session_state['readiness_decision']}**")

            st.markdown(f"---")
            st.markdown(f"### Reports and Audit Artifacts")
            st.markdown(f"Reports are generated and saved to: `{st.session_state['reports_folder_path']}`")
            st.markdown(f"All audit artifacts are bundled into: `{st.session_state['zip_archive_path']}`")

            if st.session_state['zip_archive_path'] and os.path.exists(st.session_state['zip_archive_path']):
                with open(st.session_state['zip_archive_path'], "rb") as fp:
                    st.download_button(
                        label="Download Audit Bundle (.zip)",
                        data=fp.read(),
                        file_name=os.path.basename(st.session_state['zip_archive_path']),
                        mime="application/zip",
                        key="download_zip_btn"
                    )

            st.subheader("Executive Summary:")
            if st.session_state['reports_folder_path']:
                summary_file_path = os.path.join(st.session_state['reports_folder_path'], 'session04_executive_summary.md')
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r') as f:
                        summary_content = f.read()
                    st.markdown(summary_content)
                else:
                    st.error("Executive Summary file not found.")
            else:
                st.info("Reports folder path not set. Please generate reports first.")

            st.markdown(f"---")
            st.markdown(f"### Explanation of Execution")
            st.markdown(f"Maya has now generated a comprehensive suite of reports. The `overall_dataset_readiness_status` provides a definitive answer for her stakeholders. For Software Innovators Inc., a 'DO NOT DEPLOY' status (due to the simulated 'FAIL' conditions in our sample data) means Maya must halt the model development pipeline. She then shares the `session04_executive_summary.md` with the Model Validator and Risk/Compliance Partner. This markdown document provides a high-level overview, highlighting critical issues (e.g., \"Feature `income`: Missingness: 25.00% (`FAIL`)\", \"Sensitive Attribute `region`: Disparate Impact Ratio: 0.6700 (`FAIL`)\"), and the clear recommendation.")
            st.markdown(f"The JSON reports (e.g., `data_quality_report.json`) provide the granular details necessary for deeper investigation by the technical team. This structured reporting ensures that Software Innovators Inc. maintains a clear audit trail for all data-related decisions and adheres to its internal governance framework.")

            st.markdown(f"## 6. Exporting Audit Artifacts: Ensuring Traceability and Compliance")
            st.markdown(f"For Software Innovators Inc., regulatory compliance and internal auditability are paramount. Every decision related to model development, especially data quality, must be fully traceable and defensible. Maya's final, critical step is to bundle all generated reports, configurations, and an evidence manifest (containing SHA-256 hashes of all artifacts) into a secure, version-controlled zip archive. This ensures that the assessment results are immutable, tamper-evident, and readily available for future audits or reviews by risk and compliance teams.")
            st.markdown(f"**Evidence Manifest:** A record of all generated files, including their SHA-256 hash. The SHA-256 hash is a cryptographic checksum that uniquely identifies the content of a file. If even a single bit in the file changes, its SHA-256 hash will be drastically different.")
            st.markdown(r"$$ \text{{SHA-256 Hash}} = \text{{SHA256}}(\text{{file\_content}}) $$")
            st.markdown(r"where SHA-256 Hash is the cryptographic checksum, and file\_content is the entire binary content of the file.")
            st.markdown(f"This cryptographic integrity check confirms that the reports have not been altered since their generation.")
            
            st.markdown(f"---")
            st.markdown(f"### Explanation of Execution")
            st.markdown(f"Maya has successfully created a zip archive containing all the assessment reports, the configuration snapshot, and a crucial `evidence_manifest.json` file. This manifest acts as a digital fingerprint for all generated documents. The verification step demonstrates how a Model Validator or Risk Partner at Software Innovators Inc. can confirm the integrity of any report by recomputing its hash and comparing it against the manifest. If any report were to be tampered with, the hash verification would fail, immediately flagging a potential issue.")
            st.markdown(f"This robust export mechanism ensures that Software Innovators Inc. can confidently demonstrate compliance, provide auditable evidence of data quality and fairness assessments, and maintain strict governance over its ML models throughout their lifecycle. Maya's work is complete, providing Software Innovators Inc. with the objective evidence needed to make informed decisions about its credit approval model.")
```
