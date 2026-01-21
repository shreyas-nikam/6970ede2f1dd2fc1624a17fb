
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import patch, mock_open
import os
import zipfile
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns

# Mock the source.py module and its functions
class MockSource:
    def load_and_configure_datasets(primary_path, target_col, sensitive_cols, baseline_path=None, config_thresholds=None):
        primary_df = pd.read_csv(StringIO("col1,col2,target,sensitive_attr1,sensitive_attr2,numeric_col\n1,A,0,Group1,SubA,10.5\n2,B,1,Group2,SubB,20.1\n3,A,0,Group1,SubA,15.2\n4,C,1,Group1,SubC,30.0\n5,B,0,Group2,SubA,12.3\n"))
        baseline_df = pd.read_csv(StringIO("col1,col2,target,sensitive_attr1,sensitive_attr2,numeric_col\n1,A,0,Group1,SubA,11.0\n2,B,1,Group2,SubB,21.0\n3,A,0,Group1,SubA,16.0\n4,C,1,Group1,SubC,31.0\n5,B,0,Group2,SubA,13.0\n")) if baseline_path else None
        assessment_config = {
            'target_column': target_col,
            'sensitive_attributes': sensitive_cols,
            'protected_groups': config_thresholds.get('protected_groups', {}),
            'data_quality_thresholds': config_thresholds.get('data_quality_thresholds', {
                'missingness_ratio': {'warn': 0.05, 'fail': 0.20},
                'duplicate_rows_ratio': {'warn': 0.01, 'fail': 0.05},
                'type_inconsistency_ratio': {'warn': 0.00, 'fail': 0.00},
                'range_violation_ratio': {'warn': 0.00, 'fail': 0.00},
                'cardinality_unique_count_min': {'warn': 2, 'fail': 1},
                'cardinality_unique_count_max_ratio': {'warn': 0.5, 'fail': 0.9},
            }),
            'bias_thresholds': config_thresholds.get('bias_thresholds', {
                'demographic_parity_difference': {'warn': 0.10, 'fail': 0.20},
                'disparate_impact_ratio': {'warn_lower': 0.80, 'warn_upper': 1.25, 'fail_lower': 0.67, 'fail_upper': 1.50},
                'proxy_tpr_gap': {'warn': 0.10, 'fail': 0.20},
                'proxy_fpr_gap': {'warn': 0.10, 'fail': 0.20},
            }),
            'drift_thresholds': config_thresholds.get('drift_thresholds', {'psi': {'warn': 0.10, 'fail': 0.25}}),
            'feature_range_expectations': config_thresholds.get('feature_range_expectations', {})
        }
        return primary_df, baseline_df, assessment_config

    def perform_data_quality_checks(df, config):
        return {
            'overall_dataset_quality_status': 'PASS',
            'col1': {'missing_ratio': 0.0, 'missing_status': 'PASS', 'type_inconsistency_ratio': 0.0, 'consistent_type': 'int64', 'type_consistency_status': 'PASS', 'range_violation_ratio': 0.0, 'range_violation_status': 'N/A', 'unique_values_count': df['col1'].nunique(), 'cardinality_status': 'PASS'},
            'col2': {'missing_ratio': 0.0, 'missing_status': 'PASS', 'type_inconsistency_ratio': 0.0, 'consistent_type': 'object', 'type_consistency_status': 'PASS', 'range_violation_ratio': 0.0, 'range_violation_status': 'N/A', 'unique_values_count': df['col2'].nunique(), 'cardinality_status': 'PASS'},
            'target': {'missing_ratio': 0.0, 'missing_status': 'PASS', 'type_inconsistency_ratio': 0.0, 'consistent_type': 'int64', 'type_consistency_status': 'PASS', 'range_violation_ratio': 0.0, 'range_violation_status': 'N/A', 'unique_values_count': df['target'].nunique(), 'cardinality_status': 'PASS'},
            'sensitive_attr1': {'missing_ratio': 0.0, 'missing_status': 'PASS', 'type_inconsistency_ratio': 0.0, 'consistent_type': 'object', 'type_consistency_status': 'PASS', 'range_violation_ratio': 0.0, 'range_violation_status': 'N/A', 'unique_values_count': df['sensitive_attr1'].nunique(), 'cardinality_status': 'PASS'},
            'sensitive_attr2': {'missing_ratio': 0.0, 'missing_status': 'PASS', 'type_inconsistency_ratio': 0.0, 'consistent_type': 'object', 'type_consistency_status': 'PASS', 'range_violation_ratio': 0.0, 'range_violation_status': 'N/A', 'unique_values_count': df['sensitive_attr2'].nunique(), 'cardinality_status': 'PASS'},
            'numeric_col': {'missing_ratio': 0.0, 'missing_status': 'PASS', 'type_inconsistency_ratio': 0.0, 'consistent_type': 'float64', 'type_consistency_status': 'PASS', 'range_violation_ratio': 0.0, 'range_violation_status': 'PASS', 'range_min_expected': 0.0, 'range_max_expected': 100.0, 'unique_values_count': df['numeric_col'].nunique(), 'cardinality_status': 'PASS'},
            'dataset_overall': {'duplicate_rows_ratio': 0.0, 'duplicate_rows_status': 'PASS'}
        }

    def compute_bias_metrics(df, target_col, sensitive_cols, protected_groups, config):
        return {
            'overall_bias_status': 'PASS',
            'sensitive_attr1': {
                'demographic_parity_difference': 0.05, 'demographic_parity_difference_status': 'PASS',
                'disparate_impact_ratio': 1.05, 'disparate_impact_ratio_status': 'PASS',
                'proxy_tpr_gap': 0.02, 'proxy_tpr_gap_status': 'PASS',
                'proxy_fpr_gap': 0.03, 'proxy_fpr_gap_status': 'PASS'
            },
            'sensitive_attr2': {
                'demographic_parity_difference': -0.03, 'demographic_parity_difference_status': 'PASS',
                'disparate_impact_ratio': 0.98, 'disparate_impact_ratio_status': 'PASS',
                'proxy_tpr_gap': -0.01, 'proxy_tpr_gap_status': 'PASS',
                'proxy_fpr_gap': -0.02, 'proxy_fpr_gap_status': 'PASS'
            }
        }

    def calculate_psi(primary_df, baseline_df, numeric_columns, config):
        return {
            'overall_drift_status': 'PASS',
            'numeric_col': {'psi': 0.08, 'status': 'PASS'}
        }

    def make_readiness_decision(dq_results, bias_results, drift_results):
        if dq_results['overall_dataset_quality_status'] == 'FAIL' or \
           bias_results['overall_bias_status'] == 'FAIL' or \
           (drift_results and drift_results.get('overall_drift_status') == 'FAIL'):
            return 'DO NOT DEPLOY'
        elif dq_results['overall_dataset_quality_status'] == 'WARN' or \
             bias_results['overall_bias_status'] == 'WARN' or \
             (drift_results and drift_results.get('overall_drift_status') == 'WARN'):
            return 'PROCEED WITH MITIGATION'
        return 'PROCEED'

    def generate_reports(dq_results, bias_results, drift_results, readiness_decision, config, primary_filename, baseline_filename):
        # Mock creating a directory and some dummy files
        reports_dir = "temp_reports/run_mock_id"
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, 'session04_executive_summary.md'), 'w') as f:
            f.write(f"# Executive Summary\n\nDecision: {readiness_decision}")
        with open(os.path.join(reports_dir, 'data_quality_report.json'), 'w') as f:
            f.write('{}')
        return reports_dir

    def bundle_artifacts(reports_folder_path, run_id):
        # Mock creating a zip file
        zip_path = f"temp_reports/audit_bundle_{run_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('dummy.txt', 'dummy content')
        return zip_path

# Patch the source module
import sys
sys.modules['source'] = MockSource

# Now import AppTest
from streamlit.testing.v1 import AppTest

# Create dummy CSV files for upload simulation
DUMMY_PRIMARY_CSV = StringIO("col1,col2,target,sensitive_attr1,sensitive_attr2,numeric_col\n1,A,0,Group1,SubA,10.5\n2,B,1,Group2,SubB,20.1\n3,A,0,Group1,SubA,15.2\n4,C,1,Group1,SubC,30.0\n5,B,0,Group2,SubA,12.3\n")
DUMMY_BASELINE_CSV = StringIO("col1,col2,target,sensitive_attr1,sensitive_attr2,numeric_col\n1,A,0,Group1,SubA,11.0\n2,B,1,Group2,SubB,21.0\n3,A,0,Group1,SubA,16.0\n4,C,1,Group1,SubC,31.0\n5,B,0,Group2,SubA,13.0\n")

# Use a mock for `_save_uploaded_file`
def mock_save_uploaded_file(uploaded_file, directory="temp_data"):
    if uploaded_file is not None:
        mock_path = os.path.join(directory, uploaded_file.name)
        # Simulate saving the file content
        with open(mock_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return mock_path
    return None

# Patch the actual _save_uploaded_file in the app script if needed,
# but for AppTest, it's often better to mock pandas.read_csv directly.
# For simplicity, we'll assume the files are 'saved' to a mock path.

@patch('pandas.read_csv', side_effect=lambda x: pd.read_csv(StringIO(DUMMY_PRIMARY_CSV.getvalue())) if 'primary' in x else pd.read_csv(StringIO(DUMMY_BASELINE_CSV.getvalue())))
@patch('os.makedirs')
@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data="Dummy file content")
@patch('zipfile.ZipFile')
@patch('hashlib.sha256')
@patch('matplotlib.pyplot.subplots', return_value=(plt.figure(), [plt.subplot(111)])) # Mock matplotlib figure
@patch('seaborn.histplot')
def test_full_app_flow(mock_sns_histplot, mock_plt_subplots, mock_hashlib, mock_zipfile, mock_open_builtin, mock_os_path_exists, mock_os_makedirs, mock_read_csv):
    """
    Tests the full end-to-end flow of the Streamlit application.
    """
    at = AppTest.from_file("app.py").run()

    # Initial state assertions
    assert at.session_state['current_page'] == "Data Upload & Configuration"
    assert "⏳ Awaiting Configuration" in at.sidebar.markdown[5].value

    # 1. Data Upload & Configuration Page
    # Simulate file uploads
    at.file_uploader(key="primary_uploader").set_uploaded_file("primary.csv", DUMMY_PRIMARY_CSV.getvalue().encode("utf-8"), "text/csv").run()
    at.file_uploader(key="baseline_uploader").set_uploaded_file("baseline.csv", DUMMY_BASELINE_CSV.getvalue().encode("utf-8"), "text/csv").run()

    # Select target and sensitive columns
    at.selectbox(key="target_col_input").set_value("target").run()
    at.multiselect(key="sensitive_cols_input").set_value(["sensitive_attr1", "sensitive_attr2"]).run()

    # Configure protected groups
    at.expander[0].open().run() # Open the "Define Protected Groups" expander
    at.text_input(key="privileged_sensitive_attr1").set_value("Group1").run()
    at.text_input(key="unprivileged_sensitive_attr1").set_value("Group2").run()
    at.text_input(key="privileged_sensitive_attr2").set_value("SubA").run()
    at.text_input(key="unprivileged_sensitive_attr2").set_value("SubB,SubC").run()

    # Apply configuration
    at.button(key="apply_config_btn").click().run()

    assert at.success[0].value == "Configuration Applied Successfully!"
    assert at.session_state['config_applied'] == True
    assert at.session_state['assessment_config'] is not None
    assert "✅ Configuration Applied" in at.sidebar.markdown[5].value

    # 2. Data Quality Page
    at.selectbox(key="page_selector").set_value("Data Quality").run()
    assert at.session_state['current_page'] == "Data Quality"

    at.button(key="run_dq_checks").click().run()
    assert at.success[0].value == "Data Quality Checks Completed!"
    assert at.session_state['data_quality_results']['overall_dataset_quality_status'] == 'PASS'
    assert "Overall Data Quality Status: PASS" in at.markdown[11].value
    assert at.dataframe[0] is not None
    assert "✅ Data Quality: **PASS**" in at.sidebar.markdown[7].value

    # 3. Bias Metrics Page
    at.selectbox(key="page_selector").set_value("Bias Metrics").run()
    assert at.session_state['current_page'] == "Bias Metrics"

    at.button(key="run_bias_checks").click().run()
    assert at.success[0].value == "Bias Metrics Computed!"
    assert at.session_state['bias_metrics_results']['overall_bias_status'] == 'PASS'
    assert "Overall Bias Status: PASS" in at.markdown[16].value
    assert at.dataframe[0] is not None
    assert "✅ Bias Metrics: **PASS**" in at.sidebar.markdown[8].value

    # 4. Drift Page
    at.selectbox(key="page_selector").set_value("Drift").run()
    assert at.session_state['current_page'] == "Drift"

    at.button(key="run_drift_checks").click().run()
    assert at.success[0].value == "Drift Detection Completed!"
    assert at.session_state['drift_detection_results']['overall_drift_status'] == 'PASS'
    assert "Overall Drift Status: PASS" in at.markdown[15].value
    assert at.dataframe[0] is not None
    assert "✅ Drift: **PASS**" in at.sidebar.markdown[9].value
    assert mock_sns_histplot.called # Verify that plotting function was called

    # 5. Summary & Export Page
    at.selectbox(key="page_selector").set_value("Summary & Export").run()
    assert at.session_state['current_page'] == "Summary & Export"

    at.button(key="generate_reports_btn").click().run()
    assert at.success[0].value == "Reports Generated and Artifacts Bundled!"
    assert at.session_state['readiness_decision'] == 'PROCEED'
    assert "## **PROCEED**" in at.markdown[8].value
    assert at.session_state['reports_folder_path'] is not None
    assert at.session_state['zip_archive_path'] is not None
    assert at.download_button[0] is not None # Verify download button exists
    assert "Final Decision: **PROCEED**" in at.sidebar.markdown[10].value

    # Test without baseline in drift page
    at_no_baseline = AppTest.from_file("app.py").run()
    at_no_baseline.file_uploader(key="primary_uploader").set_uploaded_file("primary.csv", DUMMY_PRIMARY_CSV.getvalue().encode("utf-8"), "text/csv").run()
    at_no_baseline.selectbox(key="target_col_input").set_value("target").run()
    at_no_baseline.multiselect(key="sensitive_cols_input").set_value(["sensitive_attr1"]).run()
    at_no_baseline.expander[0].open().run()
    at_no_baseline.text_input(key="privileged_sensitive_attr1").set_value("Group1").run()
    at_no_baseline.text_input(key="unprivileged_sensitive_attr1").set_value("Group2").run()
    at_no_baseline.button(key="apply_config_btn").click().run()

    at_no_baseline.selectbox(key="page_selector").set_value("Drift").run()
    assert "Baseline dataset not provided. Drift detection will be skipped." in at_no_baseline.info[0].value
    assert at_no_baseline.session_state['drift_detection_results']['overall_drift_status'] == 'N/A'
    assert "⚠️ Drift: No Baseline Provided" in at_no_baseline.sidebar.markdown[9].value

    # Test configuration warnings
    at_config_warn = AppTest.from_file("app.py").run()
    at_config_warn.file_uploader(key="primary_uploader").set_uploaded_file("primary.csv", DUMMY_PRIMARY_CSV.getvalue().encode("utf-8"), "text/csv").run()
    at_config_warn.button(key="apply_config_btn").click().run() # No target/sensitive selected
    assert at_config_warn.warning[0].value == "Please upload a primary dataset, select a target column, and sensitive attributes to apply configuration."

