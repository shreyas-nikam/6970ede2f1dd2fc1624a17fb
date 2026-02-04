import streamlit as st
import pandas as pd
import json
import os
import datetime
import zipfile
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import uuid
from source import *
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="QuLab: Lab 4: Data Quality, Provenance & Bias Metrics Dashboard", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 4: Data Quality, Provenance & Bias Metrics Dashboard")
st.divider()

# Helper function to generate unique session directory


def _get_unique_session_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"temp_data/session_{timestamp}_{unique_id}"

# Helper function to save uploaded files to unique session directory


def _save_uploaded_file(uploaded_file, session_dir):
    if uploaded_file is not None:
        os.makedirs(session_dir, exist_ok=True)
        file_path = os.path.join(session_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# Helper function to cleanup session files


def _cleanup_session_files():
    """Delete all uploaded files, reports, and reset session state"""
    try:
        # Delete uploaded files directory
        if st.session_state.get('session_dir') and os.path.exists(st.session_state['session_dir']):
            shutil.rmtree(st.session_state['session_dir'])

        # Delete reports folder
        if st.session_state.get('reports_folder_path') and os.path.exists(st.session_state['reports_folder_path']):
            shutil.rmtree(st.session_state['reports_folder_path'])

        # Delete zip file if in a different location
        if st.session_state.get('zip_archive_path') and os.path.exists(st.session_state['zip_archive_path']):
            parent_dir = os.path.dirname(st.session_state['zip_archive_path'])
            if parent_dir != st.session_state.get('reports_folder_path'):
                os.remove(st.session_state['zip_archive_path'])

        # Reset relevant session state variables
        st.session_state['primary_df'] = None
        st.session_state['baseline_df'] = None
        st.session_state['assessment_config'] = None
        st.session_state['data_quality_results'] = None
        st.session_state['bias_metrics_results'] = None
        st.session_state['drift_detection_results'] = {
            'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
        st.session_state['readiness_decision'] = 'Pending Configuration'
        st.session_state['reports_folder_path'] = None
        st.session_state['zip_archive_path'] = None
        st.session_state['primary_data_filename'] = None
        st.session_state['baseline_data_filename'] = None
        st.session_state['primary_temp_path'] = None
        st.session_state['baseline_temp_path'] = None
        st.session_state['config_applied'] = False
        st.session_state['sample_loaded'] = False
        st.session_state['session_dir'] = None
        st.session_state['cleanup_triggered'] = False

        return True
    except Exception as e:
        st.error(f"Error during cleanup: {e}")
        return False


# Session State Initialization
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
    st.session_state['drift_detection_results'] = {
        'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
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
if 'selected_use_case' not in st.session_state:
    st.session_state['selected_use_case'] = 'credit'
if 'sample_loaded' not in st.session_state:
    st.session_state['sample_loaded'] = False
if not st.session_state.get('session_dir'):
    st.session_state['session_dir'] = _get_unique_session_dir()
if 'cleanup_triggered' not in st.session_state:
    st.session_state['cleanup_triggered'] = False

# Sidebar Navigation
with st.sidebar:
    st.markdown(f"# Data Risk Assessment")
    st.markdown(f"**Persona:** Maya, Senior ML Engineer")
    st.markdown(f"---")
    st.session_state['current_page'] = st.selectbox(
        "Choose an Assessment Step:",
        ["Data Upload & Configuration", "Data Quality",
            "Bias Metrics", "Drift", "Summary & Export"],
        key="page_selector"
    )
    st.markdown(f"---")
    st.markdown(f"## Current Status:")
    if st.session_state.get('config_applied', False):
        st.markdown(f"✅ Configuration Applied")
    else:
        st.markdown(f"⏳ Awaiting Configuration")
    if st.session_state.get('data_quality_results') is not None:
        st.markdown(
            f"✅ Data Quality: **{st.session_state['data_quality_results']['overall_dataset_quality_status']}**")
    else:
        st.markdown(f"⏳ Data Quality Pending")
    if st.session_state.get('bias_metrics_results') is not None:
        st.markdown(
            f"✅ Bias Metrics: **{st.session_state['bias_metrics_results']['overall_bias_status']}**")
    else:
        st.markdown(f"⏳ Bias Metrics Pending")
    if st.session_state.get('baseline_df') is not None and st.session_state.get('drift_detection_results') is not None:
        st.markdown(
            f"✅ Drift: **{st.session_state['drift_detection_results']['overall_drift_status']}**")
    elif st.session_state.get('baseline_df') is None and st.session_state.get('config_applied'):
        st.markdown(f"⚠️ Drift: No Baseline Provided")
    else:
        st.markdown(f"⏳ Drift Pending")
    if st.session_state.get('readiness_decision') != 'Pending Configuration':
        st.markdown(
            f"**Final Decision:** **{st.session_state['readiness_decision']}**")
    else:
        st.markdown(f"⏳ Final Decision Pending")

# Page: Data Upload & Configuration
if st.session_state['current_page'] == "Data Upload & Configuration":
    st.markdown(f"# 1. Setting the Stage: Data Loading and Configuration")
    st.markdown(f"Welcome to this hands-on lab designed for ML Engineers, Model Validators, and Risk/Compliance Partners. In this notebook, you will step into the role of a Senior ML Engineer at a leading technology company. Your team is responsible for developing and maintaining robust machine learning models that must meet high standards for data quality, fairness, and compliance.")
    st.markdown(f"Today, you are tasked with preparing a new dataset for a machine learning project. Before your organization commits to model training and deployment, you need to perform a comprehensive data quality and risk assessment. Your goal is to ensure the raw data meets fundamental quality standards, is free from hidden biases, and has not drifted significantly from historical data, thereby reducing model risk and supporting responsible AI practices.")
    st.markdown(f"This section guides you through loading your new dataset and configuring parameters for a tailored assessment. Defining these parameters upfront is crucial to align the assessment with your organization's specific model requirements and compliance policies.")

    t1, t2 = st.tabs(["Upload Dataset", "Load Sample"])
    with t2:
        st.subheader("Select Use Case Scenario")

        # Use case selection
        use_case_options = {
            'credit': 'Credit Approval Dataset\n\nCustomer-level tabular data with protected attributes.',
            'healthcare': 'Healthcare Outcomes Dataset\n\nPatient triage or diagnosis support data.',
            'fraud': 'Operations / Fraud Dataset\n\nHigh-volume transactional data with temporal drift.'
        }

        selected_use_case = st.radio(
            "",
            options=list(use_case_options.keys()),
            format_func=lambda x: use_case_options[x],
            key='use_case_radio',
            index=list(use_case_options.keys()).index(
                st.session_state.get('selected_use_case', 'credit'))
        )
        st.session_state['selected_use_case'] = selected_use_case

        st.markdown("###  ")
        if st.button("Load Sample Dataset", key="load_sample_btn", width='stretch'):
            with st.spinner(f"Loading sample {selected_use_case} dataset..."):
                try:
                    primary_df, baseline_df, config_defaults = setup_sample_data(
                        use_case=selected_use_case)
                    st.session_state['primary_df'] = primary_df
                    # Copy sample files to unique session directory
                    session_dir = st.session_state['session_dir']
                    os.makedirs(session_dir, exist_ok=True)

                    # Copy primary file
                    primary_session_path = os.path.join(
                        session_dir, os.path.basename(config_defaults['primary_path']))
                    shutil.copy2(
                        config_defaults['primary_path'], primary_session_path)

                    # Copy baseline file
                    baseline_session_path = os.path.join(
                        session_dir, os.path.basename(config_defaults['baseline_path']))
                    shutil.copy2(
                        config_defaults['baseline_path'], baseline_session_path)

                    # Update session state with loaded data
                    st.session_state['primary_df'] = primary_df
                    st.session_state['baseline_df'] = baseline_df
                    st.session_state['primary_temp_path'] = primary_session_path
                    st.session_state['baseline_temp_path'] = baseline_session_path
                    st.session_state['primary_data_filename'] = os.path.basename(
                        primary_session_path)
                    st.session_state['baseline_data_filename'] = os.path.basename(
                        baseline_session_path)
                    st.session_state['sample_loaded'] = True

                    # Store suggested configuration
                    st.session_state['suggested_target'] = config_defaults['target_column']
                    st.session_state['suggested_sensitive'] = config_defaults['sensitive_attributes']
                    st.session_state['suggested_protected_groups'] = config_defaults['protected_groups']

                    st.success(
                        f"✅ Sample {selected_use_case} dataset loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading sample dataset: {e}")

        # Display loaded sample datasets with interactive exploration
        if st.session_state.get('sample_loaded', False) and st.session_state['primary_df'] is not None:
            st.markdown(f"**📊 Interactive Dataset Explorer**")

            explore_tab1, explore_tab2, explore_tab3 = st.tabs(
                ["Data Preview", "Column Statistics", "Data Quality Quick View"])

            with explore_tab1:
                # Interactive data preview with filters
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    search_col = st.selectbox("Filter by column:", [
                                              "All"] + list(st.session_state['primary_df'].columns), key="sample_search_col")
                with col2:
                    num_rows = st.slider("Rows to display:", 5, min(
                        100, len(st.session_state['primary_df'])), 10, key="sample_rows")
                with col3:
                    sort_col = st.selectbox("Sort by:", [
                                            "None"] + list(st.session_state['primary_df'].columns), key="sample_sort")

                df_display = st.session_state['primary_df'].copy()
                if sort_col != "None":
                    df_display = df_display.sort_values(sort_col)

                st.dataframe(df_display.head(num_rows),
                             width='stretch', height=400)
                st.caption(
                    f"Showing {num_rows} of {len(st.session_state['primary_df'])} rows")

            with explore_tab2:
                # Column statistics
                col_to_analyze = st.selectbox(
                    "Select column to analyze:", st.session_state['primary_df'].columns, key="sample_col_analyze")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Basic Statistics**")
                    if pd.api.types.is_numeric_dtype(st.session_state['primary_df'][col_to_analyze]):
                        stats = st.session_state['primary_df'][col_to_analyze].describe(
                        )
                        st.dataframe(stats)
                    else:
                        value_counts = st.session_state['primary_df'][col_to_analyze].value_counts(
                        )
                        st.dataframe(value_counts)

                with col2:
                    st.markdown("**Distribution Visualization**")
                    if pd.api.types.is_numeric_dtype(st.session_state['primary_df'][col_to_analyze]):
                        fig = plt.figure(figsize=(6, 4))
                        st.session_state['primary_df'][col_to_analyze].hist(
                            bins=30, edgecolor='black')
                        plt.xlabel(col_to_analyze)
                        plt.ylabel('Frequency')
                        plt.title(f'Distribution of {col_to_analyze}')
                        st.pyplot(fig)
                        plt.close()
                    else:
                        fig = plt.figure(figsize=(6, 4))
                        st.session_state['primary_df'][col_to_analyze].value_counts().head(
                            10).plot(kind='bar')
                        plt.xlabel(col_to_analyze)
                        plt.ylabel('Count')
                        plt.title(f'Top 10 Values in {col_to_analyze}')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

            with explore_tab3:
                st.markdown("**Quick Data Quality Overview**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    missing_pct = (st.session_state['primary_df'].isnull(
                    ).sum() / len(st.session_state['primary_df']) * 100)
                    st.metric("Columns with Missing Data",
                              f"{(missing_pct > 0).sum()}/{len(st.session_state['primary_df'].columns)}")
                with col2:
                    duplicates = st.session_state['primary_df'].duplicated(
                    ).sum()
                    st.metric("Duplicate Rows", duplicates)
                with col3:
                    numeric_cols = st.session_state['primary_df'].select_dtypes(
                        include=np.number).columns
                    st.metric("Numeric Columns", len(numeric_cols))

                # Missing data heatmap
                st.markdown("**Missing Data Pattern**")
                missing_data = st.session_state['primary_df'].isnull().sum()
                if missing_data.sum() > 0:
                    fig = plt.figure(figsize=(10, 4))
                    missing_data[missing_data > 0].plot(
                        kind='bar', color='coral')
                    plt.xlabel('Columns')
                    plt.ylabel('Missing Values Count')
                    plt.title('Missing Values by Column')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.success("✅ No missing data detected!")

            if st.session_state['baseline_df'] is not None:
                st.divider()
                st.markdown(f"**📊 Baseline Dataset Preview**")
                st.dataframe(st.session_state['baseline_df'].head(
                    10), width='stretch')

        st.divider()

    with t1:
        st.subheader("Upload Datasets")
        primary_uploaded_file = st.file_uploader("Upload Primary Dataset (CSV)", type=[
            "csv"], key="primary_uploader")
        baseline_uploaded_file = st.file_uploader(
            "Upload Optional Baseline Dataset (CSV) for Drift Detection", type=["csv"], key="baseline_uploader")

    # Handle uploaded files, but don't clear if sample data was loaded
    if primary_uploaded_file:
        st.session_state['primary_data_filename'] = primary_uploaded_file.name
        st.session_state['primary_temp_path'] = _save_uploaded_file(
            primary_uploaded_file, st.session_state['session_dir'])
        st.session_state['primary_df'] = pd.read_csv(
            st.session_state['primary_temp_path'])
        # User uploaded their own file
        st.session_state['sample_loaded'] = False
        st.markdown(f"**Primary Dataset Preview:**")
        st.dataframe(st.session_state['primary_df'].head())
    elif not st.session_state.get('sample_loaded', False):
        # Only clear if no sample was loaded
        st.session_state['primary_df'] = None
        st.session_state['primary_data_filename'] = None
        st.session_state['primary_temp_path'] = None

    if baseline_uploaded_file:
        st.session_state['baseline_data_filename'] = baseline_uploaded_file.name
        st.session_state['baseline_temp_path'] = _save_uploaded_file(
            baseline_uploaded_file, st.session_state['session_dir'])
        st.session_state['baseline_df'] = pd.read_csv(
            st.session_state['baseline_temp_path'])
        # User uploaded their own file
        st.session_state['sample_loaded'] = False
        st.markdown(f"**Baseline Dataset Preview:**")
        st.dataframe(st.session_state['baseline_df'].head())
    elif not st.session_state.get('sample_loaded', False):
        # Only clear if no sample was loaded
        st.session_state['baseline_df'] = None
        st.session_state['baseline_data_filename'] = None
        st.session_state['baseline_temp_path'] = None

    if st.session_state['primary_df'] is not None:
        all_columns = st.session_state['primary_df'].columns.tolist()

        st.subheader("Configure Assessment Parameters")

        # Use suggested values if sample was loaded, otherwise use defaults
        default_target = st.session_state.get(
            'suggested_target', all_columns[0] if all_columns else None)
        default_sensitive = st.session_state.get('suggested_sensitive', [])

        target_col_input = st.selectbox(
            "Select Target Label Column",
            all_columns,
            index=all_columns.index(
                default_target) if default_target in all_columns else 0,
            key="target_col_input"
        )
        sensitive_cols_input = st.multiselect(
            "Select Sensitive Attributes (for Bias Detection)",
            all_columns,
            default=default_sensitive if all(
                attr in all_columns for attr in default_sensitive) else [],
            key="sensitive_cols_input"
        )

        if sensitive_cols_input:
            st.markdown(
                f"A key concern is ensuring fairness. The selected sensitive attributes `{', '.join(sensitive_cols_input)}` will be analyzed for potential biases.")
        else:
            st.markdown(
                f"Select sensitive attributes to enable bias detection in your dataset.")

        # Protected Groups configuration
        with st.expander("Define Protected Groups (e.g., Privileged/Unprivileged)"):
            st.markdown(
                f"Specify the privileged and unprivileged groups for each sensitive attribute. These definitions are crucial for bias metric computation.")
            configured_protected_groups = {}
            suggested_groups = st.session_state.get(
                'suggested_protected_groups', {})

            for attr in sensitive_cols_input:
                st.markdown(f"**Sensitive Attribute: `{attr}`**")

                # Get suggested values or use smart defaults
                if attr in suggested_groups:
                    privileged_default = suggested_groups[attr]['privileged']
                    unprivileged_default = ','.join(
                        suggested_groups[attr]['unprivileged'])
                elif attr == "marital_status":
                    privileged_default = "Married"
                    unprivileged_default = "Single,Divorced"
                elif attr == "region":
                    privileged_default = "North"
                    unprivileged_default = "South,East,West"
                elif attr == "ethnicity":
                    privileged_default = "White"
                    unprivileged_default = "Black,Hispanic,Asian"
                elif attr == "gender":
                    privileged_default = "Male"
                    unprivileged_default = "Female"
                elif attr == "device_type":
                    privileged_default = "Desktop"
                    unprivileged_default = "Mobile,Tablet"
                else:
                    privileged_default = ""
                    unprivileged_default = ""

                privileged_group = st.text_input(
                    f"Privileged group for `{attr}`:",
                    value=privileged_default,
                    key=f"privileged_{attr}"
                )
                unprivileged_groups_str = st.text_input(
                    f"Unprivileged groups for `{attr}` (comma-separated):",
                    value=unprivileged_default,
                    key=f"unprivileged_{attr}"
                )

                if privileged_group and unprivileged_groups_str:
                    configured_protected_groups[attr] = {
                        'privileged': privileged_group,
                        'unprivileged': [g.strip() for g in unprivileged_groups_str.split(',')]
                    }
                else:
                    st.warning(
                        f"Please define both privileged and unprivileged groups for `{attr}` if you wish to include it in bias analysis.")

            if not sensitive_cols_input:
                st.info(
                    "No sensitive attributes selected. Protected groups configuration is optional.")
            elif not configured_protected_groups:
                st.warning(
                    "No protected groups configured. Bias metrics may not be computed correctly.")

        # Threshold Overrides
        custom_thresholds = {}
        with st.expander("Override Default Thresholds"):
            st.markdown(f"Maya can override predefined warning and failure thresholds for data quality, bias, and drift metrics to align with Software Innovators Inc.'s specific risk appetite.")
            st.markdown(f"### Data Quality Thresholds")
            st.markdown(f"**Missingness Ratio:** (> value WARN, > value FAIL)")
            dq_missing_warn = st.number_input(
                "Missingness WARN Ratio", value=0.05, format="%.2f", key="dq_missing_warn")
            dq_missing_fail = st.number_input(
                "Missingness FAIL Ratio", value=0.20, format="%.2f", key="dq_missing_fail")
            custom_thresholds['data_quality_thresholds'] = {
                'missingness_ratio': {'warn': dq_missing_warn, 'fail': dq_missing_fail}}

            st.markdown(
                f"**Duplicate Rows Ratio:** (> value WARN, > value FAIL)")
            dq_dup_warn = st.number_input(
                "Duplicate Rows WARN Ratio", value=0.01, format="%.2f", key="dq_dup_warn")
            dq_dup_fail = st.number_input(
                "Duplicate Rows FAIL Ratio", value=0.05, format="%.2f", key="dq_dup_fail")
            custom_thresholds['data_quality_thresholds']['duplicate_rows_ratio'] = {
                'warn': dq_dup_warn, 'fail': dq_dup_fail}

            st.markdown(
                f"**Type Inconsistency Ratio:** (> value WARN, > value FAIL)")
            dq_type_warn = st.number_input(
                "Type Inconsistency WARN Ratio", value=0.00, format="%.2f", key="dq_type_warn")
            dq_type_fail = st.number_input(
                "Type Inconsistency FAIL Ratio", value=0.00, format="%.2f", key="dq_type_fail")
            custom_thresholds['data_quality_thresholds']['type_inconsistency_ratio'] = {
                'warn': dq_type_warn, 'fail': dq_type_fail}

            st.markdown(
                f"**Range Violation Ratio:** (> value WARN, > value FAIL)")
            dq_range_warn = st.number_input(
                "Range Violation WARN Ratio", value=0.00, format="%.2f", key="dq_range_warn")
            dq_range_fail = st.number_input(
                "Range Violation FAIL Ratio", value=0.00, format="%.2f", key="dq_range_fail")
            custom_thresholds['data_quality_thresholds']['range_violation_ratio'] = {
                'warn': dq_range_warn, 'fail': dq_range_fail}

            st.markdown(
                f"**Cardinality Unique Count (Min):** (< value WARN, < value FAIL)")
            dq_card_min_warn = st.number_input(
                "Cardinality Min WARN Count", value=2, key="dq_card_min_warn")
            dq_card_min_fail = st.number_input(
                "Cardinality Min FAIL Count", value=1, key="dq_card_min_fail")
            custom_thresholds['data_quality_thresholds']['cardinality_unique_count_min'] = {
                'warn': dq_card_min_warn, 'fail': dq_card_min_fail}

            st.markdown(
                f"**Cardinality Unique Count (Max Ratio):** (> value WARN, > value FAIL)")
            dq_card_max_warn = st.number_input(
                "Cardinality Max WARN Ratio (e.g., 0.5 for >50% unique)", value=0.5, format="%.2f", key="dq_card_max_warn")
            dq_card_max_fail = st.number_input(
                "Cardinality Max FAIL Ratio (e.g., 0.9 for >90% unique)", value=0.9, format="%.2f", key="dq_card_max_fail")
            custom_thresholds['data_quality_thresholds']['cardinality_unique_count_max_ratio'] = {
                'warn': dq_card_max_warn, 'fail': dq_card_max_fail}

            st.markdown(f"### Feature Range Expectations (Numerical Features)")
            configured_feature_ranges = {}
            numeric_cols = st.session_state['primary_df'].select_dtypes(
                include=np.number).columns.tolist()
            for col in numeric_cols:
                col_range_expander = st.expander(f"Set Range for '{col}'")
                with col_range_expander:
                    min_val = st.number_input(
                        f"Min value for '{col}'", value=0.0 if 'income' in col or 'loan_amount' in col else 18.0 if 'age' in col else 300.0 if 'credit_score' in col else 0.0, key=f"range_min_{col}")
                    max_val = st.number_input(
                        f"Max value for '{col}'", value=500000.0 if 'income' in col else 100000.0 if 'loan_amount' in col else 90.0 if 'age' in col else 850.0 if 'credit_score' in col else 1000.0, key=f"range_max_{col}")
                    configured_feature_ranges[col] = {
                        'min': min_val, 'max': max_val}
            custom_thresholds['feature_range_expectations'] = configured_feature_ranges

            st.markdown(f"### Bias Thresholds")
            st.markdown(
                f"**Demographic Parity Difference:** (Abs > value WARN, Abs > value FAIL)")
            bias_dpd_warn = st.number_input(
                "DPD WARN Threshold", value=0.10, format="%.2f", key="bias_dpd_warn")
            bias_dpd_fail = st.number_input(
                "DPD FAIL Threshold", value=0.20, format="%.2f", key="bias_dpd_fail")
            custom_thresholds['bias_thresholds'] = {'demographic_parity_difference': {
                'warn': bias_dpd_warn, 'fail': bias_dpd_fail}}

            st.markdown(
                f"**Disparate Impact Ratio:** (< lower_value or > upper_value WARN, < lower_value or > upper_value FAIL)")
            bias_dir_warn_l = st.number_input(
                "DIR WARN Lower Bound", value=0.80, format="%.2f", key="bias_dir_warn_l")
            bias_dir_warn_u = st.number_input(
                "DIR WARN Upper Bound", value=1.25, format="%.2f", key="bias_dir_warn_u")
            bias_dir_fail_l = st.number_input(
                "DIR FAIL Lower Bound", value=0.67, format="%.2f", key="bias_dir_fail_l")
            bias_dir_fail_u = st.number_input(
                "DIR FAIL Upper Bound", value=1.50, format="%.2f", key="bias_dir_fail_u")
            custom_thresholds['bias_thresholds']['disparate_impact_ratio'] = {
                'warn_lower': bias_dir_warn_l, 'warn_upper': bias_dir_warn_u, 'fail_lower': bias_dir_fail_l, 'fail_upper': bias_dir_fail_u}

            st.markdown(f"### Drift Thresholds (PSI)")
            st.markdown(
                f"**Population Stability Index (PSI):** (> value WARN, > value FAIL)")
            drift_psi_warn = st.number_input(
                "PSI WARN Threshold", value=0.10, format="%.2f", key="drift_psi_warn")
            drift_psi_fail = st.number_input(
                "PSI FAIL Threshold", value=0.25, format="%.2f", key="drift_psi_fail")
            custom_thresholds['drift_thresholds'] = {
                'psi': {'warn': drift_psi_warn, 'fail': drift_psi_fail}}

        col1, col2 = st.columns([1, 1])
        with col1:
            apply_config_clicked = st.button(
                "Apply Configuration", key="apply_config_btn", width='stretch')
        with col2:
            cleanup_clicked = st.button(
                "🗑️ Clear All Data & Reset", key="cleanup_btn", width='stretch', type="secondary")

        if cleanup_clicked:
            if _cleanup_session_files():
                st.success(
                    "All data cleared successfully! The page will refresh.")
                st.rerun()
            else:
                st.error("Failed to clear all data. Some files may remain.")

        if apply_config_clicked:
            if st.session_state['primary_df'] is not None and target_col_input and sensitive_cols_input:
                try:
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
                    st.session_state['data_quality_results'] = None
                    st.session_state['bias_metrics_results'] = None
                    st.session_state['drift_detection_results'] = {
                        'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
                    st.session_state['readiness_decision'] = 'Pending Configuration'
                except Exception as e:
                    st.error(f"Error applying configuration: {e}")
                    st.session_state['config_applied'] = False
            else:
                st.warning(
                    "Please upload a primary dataset, select a target column, and sensitive attributes to apply configuration.")

        if st.session_state['assessment_config']:
            st.subheader("✅ Current Assessment Configuration Applied")

            config = st.session_state['assessment_config']

            with st.expander("View Configuration Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Target Column:**")
                    st.code(config['target_column'])
                    st.markdown("**Sensitive Attributes:**")
                    for attr in config['sensitive_attributes']:
                        st.code(attr)
                with col2:
                    st.markdown("**Protected Groups:**")
                    for attr, groups in config['protected_groups'].items():
                        st.markdown(f"*{attr}:*")
                        st.text(f"  Privileged: {groups['privileged']}")
                        st.text(f"  Unprivileged: {groups['unprivileged']}")

            st.info(
                f"📊 **Dataset:** {st.session_state.get('primary_data_filename', 'Unknown')} ({len(st.session_state['primary_df'])} rows, {len(st.session_state['primary_df'].columns)} columns)")

            if st.session_state.get('baseline_df') is not None:
                st.info(
                    f"📊 **Baseline Dataset:** {st.session_state.get('baseline_data_filename', 'Unknown')} ({len(st.session_state['baseline_df'])} rows)")

            st.markdown(f"Configuration successfully applied. You can now proceed to the **Data Quality**, **Bias Metrics**, and **Drift** assessment steps using the sidebar navigation.")
    else:
        st.info(
            "Please upload a primary dataset or load one to proceed with configuration.")

# Page: Data Quality
if st.session_state['current_page'] == "Data Quality":
    st.markdown(f"# 2. Core Data Quality Assessment: Uncovering Raw Data Issues")
    st.markdown(f"Before any sophisticated modeling, Maya must ensure the fundamental quality of the dataset. This means checking for common issues like missing values, duplicate entries, inconsistent data types, values outside expected ranges, and inappropriate cardinality for categorical features. Catching these problems early prevents downstream errors in model training, improves model robustness, and saves significant computational resources. ")
    st.markdown(f"Maya will use the configured thresholds to assign a `PASS`, `WARN`, or `FAIL` status to each quality aspect of every feature.")

    st.markdown(
        r"**Missingness Ratio ($M_i$):** The proportion of missing values for feature $i$.")
    st.markdown(
        r"""
$$
M_i = \frac{{\text{{Number of Missing Values in Feature }} i}}{{\text{{Total Number of Rows}}}}
$$
""")
    st.markdown(r"where $M_i$ is the missingness ratio for feature $i$, Number of Missing Values in Feature $i$ is the count of null values in column $i$, and Total Number of Rows is the total number of entries in the dataset.")

    st.markdown(
        r"**Duplicate Rows Ratio ($D$):** The proportion of rows that are exact duplicates of other rows in the dataset.")
    st.markdown(
        r"""
$$
D = \frac{{\text{{Number of Duplicate Rows}}}}{{\text{{Total Number of Rows}}}}
$$""")
    st.markdown(r"where $D$ is the duplicate rows ratio, Number of Duplicate Rows is the count of rows that are identical to another row, and Total Number of Rows is the total number of entries in the dataset.")

    st.markdown(f"**Type Inconsistency:** Measured as the ratio of non-conforming data types within a column. For example, if a numeric column contains string values, this metric will be high.")

    st.markdown(r"**Range Violation Ratio:** For numerical features, this is the ratio of values falling outside a predefined acceptable range.")
    st.markdown(
        r"""
$$
R_i = \frac{{\text{{Number of Values Outside Expected Range for Feature }} i}}{{\text{{Total Number of Rows}}}}
$$""")
    st.markdown(r"where $R_i$ is the range violation ratio for feature $i$, Number of Values Outside Expected Range for Feature $i$ is the count of values in column $i$ that are outside the specified min/max range, and Total Number of Rows is the total number of entries in the dataset.")

    st.markdown(f"**Cardinality Check:** For categorical features, this examines the number of unique values. Extremely low cardinality (e.g., only one unique value) or extremely high cardinality (e.g., unique values approaching the total number of rows) can indicate issues.")

    if not st.session_state.get('config_applied'):
        st.warning(
            "Please go to 'Data Upload & Configuration' to upload data and apply configuration first.")
    elif st.session_state['primary_df'] is None:
        st.warning(
            "Primary dataset not loaded. Please upload data in 'Data Upload & Configuration' page.")
    else:
        if st.button("Perform Data Quality Checks", key="run_dq_checks"):
            with st.spinner("Running Data Quality Checks..."):
                st.session_state['data_quality_results'] = perform_data_quality_checks(
                    st.session_state['primary_df'], st.session_state['assessment_config'])
            st.success("Data Quality Checks Completed!")

        if st.session_state['data_quality_results']:
            st.subheader("Data Quality Assessment Results")
            overall_dq_status = st.session_state['data_quality_results']['overall_dataset_quality_status']
            st.markdown(
                f"**Overall Data Quality Status: {overall_dq_status}**")

            table_data = []
            dq_thresholds = st.session_state['assessment_config']['data_quality_thresholds']

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

                table_data.append({
                    "Feature": col,
                    "Metric": "Missingness Ratio",
                    "Value": f"{metrics['missing_ratio']:.2%}",
                    "Status": metrics['missing_status'],
                    "Threshold (Warn/Fail)": f">{dq_thresholds['missingness_ratio']['warn']:.2%} (W) / >{dq_thresholds['missingness_ratio']['fail']:.2%} (F)"
                })

                table_data.append({
                    "Feature": col,
                    "Metric": "Type Inconsistency Ratio",
                    "Value": f"{metrics['type_inconsistency_ratio']:.2%} (Consistent Type: {metrics['consistent_type']})",
                    "Status": metrics['type_consistency_status'],
                    "Threshold (Warn/Fail)": f">{dq_thresholds['type_inconsistency_ratio']['warn']:.2%} (W) / >{dq_thresholds['type_inconsistency_ratio']['fail']:.2%} (F)"
                })

                if metrics['range_violation_status'] != 'N/A':
                    table_data.append({
                        "Feature": col,
                        "Metric": "Range Violation Ratio",
                        "Value": f"{metrics['range_violation_ratio']:.2%} (Expected: {metrics.get('range_min_expected', 'N/A')}-{metrics.get('range_max_expected', 'N/A')})",
                        "Status": metrics['range_violation_status'],
                        "Threshold (Warn/Fail)": f">{dq_thresholds['range_violation_ratio']['warn']:.2%} (W) / >{dq_thresholds['range_violation_ratio']['fail']:.2%} (F)"
                    })

                table_data.append({
                    "Feature": col,
                    "Metric": "Unique Values Count",
                    "Value": f"{metrics['unique_values_count']}",
                    "Status": metrics['cardinality_status'],
                    "Threshold (Warn/Fail)": f"< {dq_thresholds['cardinality_unique_count_min']['warn']} or > {dq_thresholds['cardinality_unique_count_max_ratio']['warn']:.0%} (W) / < {dq_thresholds['cardinality_unique_count_min']['fail']} or > {dq_thresholds['cardinality_unique_count_max_ratio']['fail']:.0%} (F)"
                })

            st.dataframe(pd.DataFrame(table_data))

# Page: Bias Metrics
if st.session_state['current_page'] == "Bias Metrics":
    st.markdown(f"# 3. Bias Metric Computation: Ensuring Fairness in Data")
    st.markdown(f"At Software Innovators Inc., ensuring fairness and avoiding discrimination in credit decisions is not just a regulatory requirement but a core ethical principle. Maya understands that biases present in the training data can be learned and amplified by models, leading to unfair outcomes for certain demographic groups. Before training the credit approval model, she must quantify any inherent biases within the new dataset. This assessment helps her identify if the raw data itself exhibits disparities in credit repayment outcomes across sensitive attributes like `marital_status` or `region`.")
    st.markdown(f"Since we are in a pre-training context (no model predictions yet), we will focus on measuring statistical parity and outcome disparities based on the *actual* target variable distributions across protected groups.")

    st.markdown(r"**Demographic Parity Difference (DPD):** Measures the difference in the proportion of the favorable outcome (e.g., loan repaid) between an unprivileged group and a privileged group. A value close to 0 indicates demographic parity.")
    st.markdown(
        r"""
$$
DPD = P(Y=1 | A_{\text{unprivileged}}) - P(Y=1 | A_{\text{privileged}})
$$
""")
    st.markdown(
        r"where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.")

    st.markdown(r"**Disparate Impact Ratio (DIR):** Measures the ratio of the favorable outcome proportion for the unprivileged group to the privileged group. A value near 1 suggests no disparate impact. Values significantly below 1 (e.g., < 0.8) indicate the unprivileged group is less likely to receive the favorable outcome, while values significantly above 1 (e.g., > 1.25) indicate the unprivileged group is more likely.")
    st.markdown(
        r"""$$
DIR = \frac{{P(Y=1 | A_{\text{unprivileged}})}}{{P(Y=1 | A_{\text{privileged}})}}
$$""")
    st.markdown(
        r"where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.")

    st.info("""
    **Note on Fairness Metrics**: This assessment focuses on **pre-training fairness** using only outcome data (Y). 
    We compute:
    - **Demographic Parity Difference (DPD)**: Difference in positive outcome rates between groups
    - **Disparate Impact Ratio (DIR)**: Ratio of positive outcome rates
    
    **Not included**: TPR/FPR gaps require model predictions (Ŷ) and should be computed post-training.
    True TPR = P(Ŷ=1 | Y=1, group) and FPR = P(Ŷ=1 | Y=0, group) - these metrics assess model behavior, not data.
    """)

    if not st.session_state.get('config_applied'):
        st.warning(
            "Please go to 'Data Upload & Configuration' to upload data and apply configuration first.")
    elif st.session_state['primary_df'] is None:
        st.warning(
            "Primary dataset not loaded. Please upload data in 'Data Upload & Configuration' page.")
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

                warn_dpd = bias_thresholds['demographic_parity_difference']['warn']
                fail_dpd = bias_thresholds['demographic_parity_difference']['fail']
                table_data.append({
                    "Sensitive Attribute": attr,
                    "Metric": "Demographic Parity Difference",
                    "Value": f"{metrics['demographic_parity_difference']:.4f}",
                    "Status": metrics['demographic_parity_difference_status'],
                    "Threshold (Warn/Fail)": f"Abs > {warn_dpd:.2f} (W) / > {fail_dpd:.2f} (F)"
                })

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

            st.dataframe(pd.DataFrame(table_data))


# Page: Drift
if st.session_state['current_page'] == "Drift":
    st.markdown(f"# 4. Drift Detection: Monitoring Data Distribution Shifts")
    st.markdown(f"In any dynamic environment, the characteristics of data can change over time due to evolving behaviors, external factors, or operational changes. A machine learning model trained on older data may become less accurate if the underlying data distribution shifts—a phenomenon known as 'data drift.' To mitigate this risk, it is important to compare the new dataset against a historical 'baseline' dataset (the data the current production model was originally trained or validated on) to detect and quantify any significant changes.")

    st.markdown(r"The **Population Stability Index (PSI)** is a widely used metric to quantify data drift for numerical features. It measures how much a variable's distribution has changed from a baseline period to a current period.")
    st.markdown(
        r"**Population Stability Index (PSI):** For each feature, the PSI is calculated by:")
    st.markdown(
        r"1.  Dividing the range of the feature into several bins (e.g., 10 bins).")
    st.markdown(
        r"2.  Calculating the percentage of observations in each bin for both the baseline ($P_{{0i}}$) and the current ($P_i$) datasets.")
    st.markdown(r"3.  Summing the contributions from each bin:")
    st.markdown(
        r"$$ \text{{PSI}} = \sum_{{i=1}}^{{N}} (P_i - P_{{0i}}) \times \ln\left(\frac{{P_i}}{{P_{{0i}}}}\right) $$")
    st.markdown(
        r"where $N$ is the number of bins, $P_i$ is the percentage of observations in bin $i$ for the current dataset, and $P_{{0i}}$ is the percentage of observations in bin $i$ for the baseline dataset. A small constant (epsilon) is typically added to $P_i$ and $P_{{0i}}$ to avoid issues with zero values in the logarithm.")
    st.markdown(f"Common interpretations for PSI:")
    st.markdown(f"-   PSI < 0.1: No significant shift (PASS)")
    st.markdown(f"-   0.1 <= PSI < 0.25: Small shift (WARN)")
    st.markdown(f"-   PSI >= 0.25: Significant shift (FAIL)")

    if not st.session_state.get('config_applied'):
        st.warning(
            "Please go to 'Data Upload & Configuration' to upload data and apply configuration first.")
    elif st.session_state['primary_df'] is None:
        st.warning(
            "Primary dataset not loaded. Please upload data in 'Data Upload & Configuration' page.")
    elif st.session_state['baseline_df'] is None:
        st.info("Baseline dataset not provided. Drift detection will be skipped.")
        st.session_state['drift_detection_results'] = {
            'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
    else:
        if st.button("Detect Data Drift (PSI)", key="run_drift_checks"):
            with st.spinner("Detecting Data Drift..."):
                numeric_columns = st.session_state['primary_df'].select_dtypes(
                    include=np.number).columns.tolist()
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
                    st.session_state['drift_detection_results'] = {
                        'overall_drift_status': 'ERROR', 'message': f'Error calculating PSI: {e}'}

        if st.session_state['drift_detection_results'] and st.session_state['drift_detection_results']['overall_drift_status'] != 'N/A':
            st.subheader(
                "Drift Detection Results (Population Stability Index - PSI)")
            overall_drift_status = st.session_state['drift_detection_results']['overall_drift_status']
            st.markdown(f"**Overall Drift Status: {overall_drift_status}**")

            table_data = []
            psi_thresholds = st.session_state['assessment_config']['drift_thresholds']['psi']

            for col, metrics in st.session_state['drift_detection_results'].items():
                if col == 'overall_drift_status' or col == 'message':
                    continue

                warn_psi = psi_thresholds['warn']
                fail_psi = psi_thresholds['fail']
                psi_val_str = f"{metrics['psi']:.4f}" if not np.isnan(
                    metrics['psi']) else metrics.get('error', 'N/A')

                table_data.append({
                    "Feature": col,
                    "PSI Value": psi_val_str,
                    "Status": metrics['status'],
                    "Threshold (Warn/Fail)": f"> {warn_psi:.2f} (W) / > {fail_psi:.2f} (F)"
                })

            st.dataframe(pd.DataFrame(table_data))

            st.subheader("Visualizing Drift for Key Numerical Features")
            # Select relevant features for visualization
            features_to_plot = [
                col for col in ['age', 'income', 'credit_score', 'loan_amount']
                if col in st.session_state['primary_df'].columns and col in st.session_state['baseline_df'].columns and col in st.session_state['drift_detection_results']
            ]

            if features_to_plot:
                num_plots = len(features_to_plot)
                num_cols = 2
                num_rows = (num_plots + num_cols - 1) // num_cols

                fig, axes = plt.subplots(
                    num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
                axes = np.atleast_1d(axes).flatten()

                for i, col in enumerate(features_to_plot):
                    ax = axes[i]
                    psi_val = st.session_state['drift_detection_results'][col].get(
                        'psi', np.nan)

                    sns.histplot(st.session_state['primary_df'][col].dropna(
                    ), kde=True, color='skyblue', label='Primary', ax=ax, stat='density', alpha=0.7)
                    sns.histplot(st.session_state['baseline_df'][col].dropna(
                    ), kde=True, color='salmon', label='Baseline', ax=ax, stat='density', alpha=0.7)
                    ax.set_title(f'{col} Distribution (PSI: {psi_val:.2f})')
                    ax.legend()

                for j in range(num_plots, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(
                    "No common numerical features found or selected for drift visualization.")

        else:
            st.info(st.session_state['drift_detection_results']['message'])

# Page: Summary & Export
if st.session_state['current_page'] == "Summary & Export":
    st.markdown(f"# 5. Overall Readiness Decision and Comprehensive Reporting")
    st.markdown(f"After conducting thorough data quality, bias, and drift assessments, Maya needs to synthesize all findings into a clear, deterministic decision regarding the dataset's readiness for model training. This final decision, along with a comprehensive report, is critical for communication with Model Validators (Persona 2) and Risk/Compliance Partners (Persona 3) at Software Innovators Inc. It provides a transparent, quantitative basis for proceeding or halting model development, ensuring accountability and adherence to governance standards.")
    st.markdown(f"The overall readiness decision follows a strict logic:")
    st.markdown(f"-   **DO NOT DEPLOY:** If any 'FAIL' condition is identified across data quality, bias, or drift metrics. This indicates severe issues that must be addressed before proceeding.")
    st.markdown(f"-   **PROCEED WITH MITIGATION:** If there are only 'WARN' conditions, but no 'FAIL's. This means the dataset has minor issues that can likely be mitigated (e.g., specific data cleaning, bias-aware modeling) while proceeding with development.")
    st.markdown(f"-   **PROCEED:** If all checks result in a 'PASS'. This indicates the dataset is robust and ready for model training without immediate, significant concerns.")

    if not st.session_state.get('config_applied') or st.session_state['data_quality_results'] is None or st.session_state['bias_metrics_results'] is None:
        st.warning(
            "Please complete the 'Data Upload & Configuration', 'Data Quality', and 'Bias Metrics' steps first.")
    else:
        # Ensure drift results are initialized
        if st.session_state['drift_detection_results'] is None:
            st.session_state['drift_detection_results'] = {
                'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}

        # Display current assessment status
        st.subheader("Assessment Status Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            dq_status = st.session_state['data_quality_results']['overall_dataset_quality_status']
            if dq_status == 'FAIL':
                st.error(f"Data Quality: {dq_status}")
            elif dq_status == 'WARN':
                st.warning(f"Data Quality: {dq_status}")
            else:
                st.success(f"Data Quality: {dq_status}")
        with col2:
            bias_status = st.session_state['bias_metrics_results']['overall_bias_status']
            if bias_status == 'FAIL':
                st.error(f"Bias: {bias_status}")
            elif bias_status == 'WARN':
                st.warning(f"Bias: {bias_status}")
            else:
                st.success(f"Bias: {bias_status}")
        with col3:
            drift_status = st.session_state['drift_detection_results']['overall_drift_status']
            if drift_status == 'FAIL':
                st.error(f"Drift: {drift_status}")
            elif drift_status == 'WARN':
                st.warning(f"Drift: {drift_status}")
            elif drift_status == 'N/A':
                st.info(f"Drift: {drift_status}")
            else:
                st.success(f"Drift: {drift_status}")

        st.divider()

        if st.button("Generate Final Report and Export Artifacts", key="generate_reports_btn", width='stretch', type="primary"):
            with st.spinner("Generating reports and bundling artifacts..."):
                st.session_state['readiness_decision'] = make_readiness_decision(
                    st.session_state['data_quality_results'],
                    st.session_state['bias_metrics_results'],
                    st.session_state['drift_detection_results']
                )

                os.makedirs("data", exist_ok=True)
                os.makedirs("reports", exist_ok=True)

                st.session_state['reports_folder_path'] = generate_reports(
                    st.session_state['data_quality_results'],
                    st.session_state['bias_metrics_results'],
                    st.session_state['drift_detection_results'],
                    st.session_state['readiness_decision'],
                    st.session_state['assessment_config'],
                    st.session_state['primary_temp_path'] if st.session_state[
                        'primary_temp_path'] else st.session_state['primary_data_filename'],
                    st.session_state['baseline_temp_path'] if st.session_state[
                        'baseline_temp_path'] else st.session_state['baseline_data_filename']
                )

                run_id = os.path.basename(
                    st.session_state['reports_folder_path']).replace('run_', '')
                st.session_state['zip_archive_path'] = bundle_artifacts(
                    st.session_state['reports_folder_path'], run_id)

            st.success("Reports Generated and Artifacts Bundled!")

            # Download buttons
            st.divider()
            st.subheader("Download Reports")

            if st.session_state.get('reports_folder_path') and os.path.exists(st.session_state['reports_folder_path']):
                summary_file_path = os.path.join(
                    st.session_state['reports_folder_path'], 'session04_executive_summary.md')
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r') as f:
                        summary_content = f.read()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📄 Download Executive Summary (MD)",
                            data=summary_content,
                            file_name="executive_summary.md",
                            mime="text/markdown",
                            width='stretch'
                        )

                    # Download zip archive if available
                    if st.session_state.get('zip_archive_path') and os.path.exists(st.session_state['zip_archive_path']):
                        with col2:
                            with open(st.session_state['zip_archive_path'], 'rb') as f:
                                zip_content = f.read()
                            st.download_button(
                                label="📦 Download All Reports (ZIP)",
                                data=zip_content,
                                file_name=os.path.basename(
                                    st.session_state['zip_archive_path']),
                                mime="application/zip",
                                width='stretch'
                            )

                    st.divider()
                    st.subheader("Executive Summary Preview")
            st.markdown(f"## **{st.session_state['readiness_decision']}**")

            st.markdown(f"---")
            st.markdown(f"### Reports and Audit Artifacts")
            st.markdown(
                f"Reports are generated and saved to: `{st.session_state['reports_folder_path']}`")
            st.markdown(
                f"All audit artifacts are bundled into: `{st.session_state['zip_archive_path']}`")

            if st.session_state['zip_archive_path'] and os.path.exists(st.session_state['zip_archive_path']):
                with open(st.session_state['zip_archive_path'], "rb") as fp:
                    zip_data = fp.read()
                    download_clicked = st.download_button(
                        label="Download Audit Bundle (.zip)",
                        data=zip_data,
                        file_name=os.path.basename(
                            st.session_state['zip_archive_path']),
                        mime="application/zip",
                        key="download_zip_btn",
                        on_click=lambda: st.session_state.update(
                            {'cleanup_triggered': True})
                    )

                # Auto-cleanup after download button is clicked
                if st.session_state.get('cleanup_triggered', False):
                    if _cleanup_session_files():
                        st.success(
                            "✅ Your files have been deleted. To start again, please upload a dataset or load a sample from Page 1 (Data Upload & Configuration).")
                        st.info(
                            "🔄 The page will refresh. Please navigate to 'Data Upload & Configuration' to begin a new assessment.")
                        # Generate new session directory for next use
                        st.session_state['session_dir'] = _get_unique_session_dir(
                        )

            st.subheader("Executive Summary:")
            if st.session_state['reports_folder_path']:
                summary_file_path = os.path.join(
                    st.session_state['reports_folder_path'], 'session04_executive_summary.md')
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r') as f:
                        summary_content = f.read()
                    with st.container(border=True):
                        st.markdown(summary_content)
                else:
                    st.error("Executive Summary file not found.")
            else:
                st.info(
                    "Reports folder path not set. Please generate reports first.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
