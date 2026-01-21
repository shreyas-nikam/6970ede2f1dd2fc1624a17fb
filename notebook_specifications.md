
# Data Quality & Risk Assessment for Credit Approval Models

Welcome to this hands-on lab designed for ML Engineers, Model Validators, and Risk/Compliance Partners. In this notebook, we will step into the shoes of Maya, a Senior ML Engineer at "FinTech Innovators Inc.". Maya's team is responsible for developing and maintaining robust machine learning models that adhere to strict regulatory standards and deliver fair outcomes to customers.

Today, Maya is tasked with preparing a new dataset for a credit approval model. Before FinTech Innovators Inc. commits to costly model training and potential deployment, Maya needs to perform a comprehensive data quality and risk assessment. Her goal is to ensure the raw data meets fundamental quality standards, does not contain hidden biases, and has not drifted significantly from historical data, thereby preventing unnecessary model risk and ensuring compliance.

This notebook will guide you through Maya's workflow, demonstrating how to:
-   **Assess Data Quality:** Identify issues like missing values, duplicates, and type inconsistencies.
-   **Quantify Bias:** Evaluate potential fairness concerns related to sensitive attributes.
-   **Detect Data Drift:** Compare the new dataset's distributions against a trusted baseline.
-   **Make a Data Readiness Decision:** Determine if the dataset is fit for model training based on quantitative metrics and thresholds.
-   **Generate Audit-Ready Artifacts:** Produce comprehensive reports for internal review and regulatory compliance.

Let's begin Maya's crucial task to safeguard FinTech Innovators Inc.'s model integrity!

### Required Libraries Installation

Before we dive into the analysis, let's install all the necessary libraries.

```python
!pip install pandas numpy scipy scikit-learn tabulate matplotlib openpyxl
```

### Import Required Dependencies

Next, we'll import the Python libraries required for our data quality, bias, and drift assessment.

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import hashlib
import datetime
import os
import zipfile
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 1. Setting the Stage: Data Loading and Configuration

### Story + Context + Real-World Relevance

Maya's first step is to load the new credit application dataset and configure the parameters for the assessment. This includes identifying the target variable (what the model will predict), any sensitive attributes that need bias analysis, and setting up custom thresholds for various quality metrics. By explicitly defining these parameters upfront, Maya ensures that the assessment is tailored to FinTech Innovators Inc.'s specific model requirements and compliance policies.

For FinTech Innovators Inc., a key concern is ensuring fair lending practices. Therefore, identifying sensitive attributes like `marital_status` and `region` is paramount to later check for potential biases.

We will use synthetic datasets, `sample_credit_data.csv` (primary) and `sample_baseline_credit_data.csv` (optional baseline), designed to include some 'WARN' and 'FAIL' conditions to demonstrate the assessment capabilities.

```python
def generate_sample_data(num_rows=1000, enable_issues=True, is_baseline=False):
    """
    Generates a synthetic credit approval dataset.
    Includes intentional data quality issues if enable_issues is True.
    """
    np.random.seed(42 if not is_baseline else 100) # Different seed for baseline
    
    data = {
        'customer_id': [f'C{i:04d}' for i in range(num_rows)],
        'age': np.random.randint(18, 70, num_rows),
        'income': np.random.normal(50000, 15000, num_rows),
        'credit_score': np.random.randint(300, 850, num_rows),
        'loan_amount': np.random.normal(15000, 5000, num_rows),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], num_rows, p=[0.6, 0.2, 0.1, 0.1]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], num_rows, p=[0.4, 0.5, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_rows, p=[0.3, 0.3, 0.2, 0.2]),
        'has_other_loans': np.random.randint(0, 2, num_rows),
        'repaid_loan': np.random.randint(0, 2, num_rows) # Target variable
    }
    df = pd.DataFrame(data)

    if enable_issues and not is_baseline:
        # Introduce Missingness (FAIL) in 'income'
        missing_income_idx = np.random.choice(df.index, size=int(num_rows * 0.25), replace=False)
        df.loc[missing_income_idx, 'income'] = np.nan

        # Introduce Duplicates (WARN) in 'customer_id' and entire rows
        duplicate_ids = np.random.choice(df['customer_id'].unique(), size=int(num_rows * 0.03), replace=False)
        for _id in duplicate_ids:
            original_row = df[df['customer_id'] == _id].iloc[0]
            df = pd.concat([df, pd.DataFrame([original_row.values], columns=df.columns)], ignore_index=True)
        
        # Introduce Type Consistency (FAIL) in 'credit_score'
        # Some non-numeric entries
        type_error_idx = np.random.choice(df.index, size=int(num_rows * 0.02), replace=False)
        df.loc[type_error_idx, 'credit_score'] = 'invalid'
        
        # Introduce Range Violations (WARN) in 'age'
        range_error_idx = np.random.choice(df.index, size=int(num_rows * 0.05), replace=False)
        df.loc[range_error_idx, 'age'] = df.loc[range_error_idx, 'age'].apply(lambda x: -5 if np.random.rand() < 0.5 else 150)
        
        # Ensure 'region' has varying repayment rates for bias detection
        # Make 'South' region have a lower repayment rate
        south_idx = df[df['region'] == 'South'].index
        repaid_south_idx = np.random.choice(south_idx, size=int(len(south_idx) * 0.2), replace=False) # Only 20% repaid for 'South'
        df.loc[repaid_south_idx, 'repaid_loan'] = 0
        
    elif is_baseline:
        # For baseline, introduce slight shifts for drift detection
        df['income'] = df['income'] * np.random.normal(1.05, 0.02) # Slightly higher income on average
        df['credit_score'] = df['credit_score'] + np.random.normal(10, 5) # Slightly higher credit scores
        # Ensure baseline doesn't have extreme quality issues
        
    return df.reset_index(drop=True)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Generate and save primary dataset
primary_df = generate_sample_data(num_rows=1000, enable_issues=True, is_baseline=False)
primary_df.to_csv('data/sample_credit_data.csv', index=False)

# Generate and save baseline dataset
baseline_df = generate_sample_data(num_rows=950, enable_issues=False, is_baseline=True) # Slightly different size for baseline
baseline_df.to_csv('data/sample_baseline_credit_data.csv', index=False)

print("Sample datasets generated and saved to 'data/' directory.")

# Function to load datasets and configure parameters
def load_and_configure_datasets(primary_path, target_column, sensitive_attributes, baseline_path=None, config_thresholds=None):
    """
    Loads primary and optional baseline datasets, and sets up configuration thresholds.

    Args:
        primary_path (str): File path to the primary CSV dataset.
        target_column (str): Name of the binary target label column.
        sensitive_attributes (list): List of column names considered sensitive for bias analysis.
        baseline_path (str, optional): File path to the baseline CSV dataset for drift analysis. Defaults to None.
        config_thresholds (dict, optional): Dictionary of custom warning and failure thresholds.
                                            Defaults to None, using predefined defaults.

    Returns:
        tuple: (primary_df, baseline_df, config)
               primary_df (pd.DataFrame): The primary dataset.
               baseline_df (pd.DataFrame or None): The baseline dataset, or None if not provided.
               config (dict): A dictionary containing all configuration parameters and thresholds.
    """
    primary_df = pd.read_csv(primary_path)
    baseline_df = pd.read_csv(baseline_path) if baseline_path else None

    # Define default thresholds
    default_config = {
        'target_column': target_column,
        'sensitive_attributes': sensitive_attributes,
        'protected_groups': {
            'marital_status': {'privileged': 'Married', 'unprivileged': ['Single', 'Divorced']},
            'region': {'privileged': 'North', 'unprivileged': ['South', 'East', 'West']}
        },
        'data_quality_thresholds': {
            'missingness_ratio': {'warn': 0.05, 'fail': 0.20}, # >5% warn, >20% fail
            'duplicate_rows_ratio': {'warn': 0.01, 'fail': 0.05}, # >1% warn, >5% fail
            'type_inconsistency_ratio': {'warn': 0.0, 'fail': 0.0}, # Any inconsistency is a fail
            'range_violation_ratio': {'warn': 0.0, 'fail': 0.0}, # Any violation is a fail
            'cardinality_unique_count_min': {'warn': 2, 'fail': 1}, # <2 warn, 1 fail
            'cardinality_unique_count_max_ratio': {'warn': 0.5, 'fail': 0.9} # >50% unique warn, >90% unique fail (for categorical features, might indicate an ID column)
        },
        'feature_range_expectations': { # Example ranges for numerical features
            'age': {'min': 18, 'max': 90},
            'income': {'min': 0, 'max': 500000},
            'credit_score': {'min': 300, 'max': 850},
            'loan_amount': {'min': 0, 'max': 100000}
        },
        'bias_thresholds': {
            'demographic_parity_difference': {'warn': 0.10, 'fail': 0.20}, # |diff| > 0.10 warn, >0.20 fail
            'disparate_impact_ratio': {'warn_lower': 0.8, 'warn_upper': 1.25, 'fail_lower': 0.67, 'fail_upper': 1.5}, # Ratio outside [0.8, 1.25] warn, outside [0.67, 1.5] fail
            'proxy_tpr_gap': {'warn': 0.10, 'fail': 0.20}, # |diff| > 0.10 warn, >0.20 fail
            'proxy_fpr_gap': {'warn': 0.10, 'fail': 0.20}  # |diff| > 0.10 warn, >0.20 fail
        },
        'drift_thresholds': {
            'psi': {'warn': 0.10, 'fail': 0.25} # >0.10 warn, >0.25 fail
        }
    }

    # Override defaults with custom thresholds if provided
    if config_thresholds:
        for category, thresholds in config_thresholds.items():
            if category in default_config:
                default_config[category].update(thresholds)
    
    # Store all configurations in a single dict
    config = default_config
    
    return primary_df, baseline_df, config

# --- Execution ---
# Define paths, target, and sensitive attributes
primary_data_path = 'data/sample_credit_data.csv'
baseline_data_path = 'data/sample_baseline_credit_data.csv'
target_col = 'repaid_loan'
sensitive_cols = ['marital_status', 'region']

# Custom thresholds (example: if Maya wants to be stricter on missingness)
custom_thresholds = {
    'data_quality_thresholds': {
        'missingness_ratio': {'warn': 0.02, 'fail': 0.15}
    }
}

primary_data, baseline_data, assessment_config = load_and_configure_datasets(
    primary_data_path, 
    target_col, 
    sensitive_cols, 
    baseline_path=baseline_data_path,
    config_thresholds=custom_thresholds
)

print(f"Primary dataset loaded. Shape: {primary_data.shape}")
if baseline_data is not None:
    print(f"Baseline dataset loaded. Shape: {baseline_data.shape}")
print("\nAssessment Configuration:")
print(json.dumps(assessment_config, indent=4))
```

### Explanation of Execution

Maya has successfully loaded the datasets and reviewed the configuration. She can see the target column, the sensitive attributes, and the default (and any custom) thresholds for each metric. For instance, the stricter `missingness_ratio` thresholds she set mean that even a small percentage of missing values will trigger a 'WARN' or 'FAIL', aligning with FinTech Innovators Inc.'s high data integrity standards for financial models. This initial setup is critical to ensure the assessment is aligned with project goals and regulatory requirements.

---

## 2. Core Data Quality Assessment: Uncovering Raw Data Issues

### Story + Context + Real-World Relevance

Before any sophisticated modeling, Maya must ensure the fundamental quality of the dataset. This means checking for common issues like missing values, duplicate entries, inconsistent data types, values outside expected ranges, and inappropriate cardinality for categorical features. Catching these problems early prevents downstream errors in model training, improves model robustness, and saves significant computational resources. For FinTech Innovators Inc., poor data quality could lead to inaccurate credit risk assessments, violating internal policies and potentially regulatory guidelines.

Maya will use the configured thresholds to assign a 'PASS', 'WARN', or 'FAIL' status to each quality aspect of every feature.

**Missingness Ratio ($M_i$):** The proportion of missing values for feature $i$.
$$ M_i = \frac{\text{Number of Missing Values in Feature } i}{\text{Total Number of Rows}} $$

**Duplicate Rows Ratio ($D$):** The proportion of rows that are exact duplicates of other rows in the dataset.
$$ D = \frac{\text{Number of Duplicate Rows}}{\text{Total Number of Rows}} $$

**Type Inconsistency:** Measured as the ratio of non-conforming data types within a column. For example, if a numeric column contains string values, this metric will be high.

**Range Violation Ratio:** For numerical features, this is the ratio of values falling outside a predefined acceptable range.
$$ R_i = \frac{\text{Number of Values Outside Expected Range for Feature } i}{\text{Total Number of Rows}} $$

**Cardinality Check:** For categorical features, this examines the number of unique values. Extremely low cardinality (e.g., only one unique value) or extremely high cardinality (e.g., unique values approaching the total number of rows) can indicate issues.

```python
def perform_data_quality_checks(df, config):
    """
    Performs comprehensive data quality checks on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        config (dict): Configuration dictionary containing thresholds and feature expectations.

    Returns:
        dict: A dictionary summarizing data quality results per feature and overall.
    """
    dq_results = {}
    overall_status = 'PASS'
    dq_thresholds = config['data_quality_thresholds']
    feature_ranges = config['feature_range_expectations']

    # Overall dataset checks
    num_rows = len(df)
    
    # 1. Duplicate Rows Ratio
    duplicate_rows = df.duplicated().sum()
    duplicate_rows_ratio = duplicate_rows / num_rows if num_rows > 0 else 0
    dup_status = 'PASS'
    if duplicate_rows_ratio > dq_thresholds['duplicate_rows_ratio']['fail']:
        dup_status = 'FAIL'
        overall_status = 'FAIL'
    elif duplicate_rows_ratio > dq_thresholds['duplicate_rows_ratio']['warn']:
        dup_status = 'WARN'
        if overall_status == 'PASS': overall_status = 'WARN'

    dq_results['dataset_overall'] = {
        'duplicate_rows': duplicate_rows,
        'duplicate_rows_ratio': duplicate_rows_ratio,
        'duplicate_rows_status': dup_status
    }

    for col in df.columns:
        col_results = {}
        
        # 2. Missingness Ratio
        missing_count = df[col].isnull().sum()
        missing_ratio = missing_count / num_rows
        col_results['missing_count'] = missing_count
        col_results['missing_ratio'] = missing_ratio
        missing_status = 'PASS'
        if missing_ratio > dq_thresholds['missingness_ratio']['fail']:
            missing_status = 'FAIL'
            if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
        elif missing_ratio > dq_thresholds['missingness_ratio']['warn']:
            missing_status = 'WARN'
            if overall_status == 'PASS': overall_status = 'WARN'
        col_results['missing_status'] = missing_status

        # 3. Type Consistency
        inconsistent_types_count = 0
        consistent_type = None
        non_null_values = df[col].dropna()

        if not non_null_values.empty:
            # Determine the most common type for the column
            type_counts = non_null_values.apply(type).value_counts()
            if not type_counts.empty:
                consistent_type = type_counts.index[0]
                # Count values that do not conform to the most common type
                inconsistent_types_count = non_null_values.apply(lambda x: type(x) != consistent_type).sum()
        
        type_inconsistency_ratio = inconsistent_types_count / num_rows
        col_results['consistent_type'] = str(consistent_type)
        col_results['inconsistent_types_count'] = inconsistent_types_count
        col_results['type_inconsistency_ratio'] = type_inconsistency_ratio
        type_status = 'PASS'
        if type_inconsistency_ratio > dq_thresholds['type_inconsistency_ratio']['fail']:
            type_status = 'FAIL'
            if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
        elif type_inconsistency_ratio > dq_thresholds['type_inconsistency_ratio']['warn']:
            type_status = 'WARN'
            if overall_status == 'PASS': overall_status = 'WARN'
        col_results['type_consistency_status'] = type_status
        
        # 4. Range Violations (for numeric columns only)
        range_violation_count = 0
        range_status = 'N/A'
        if pd.api.types.is_numeric_dtype(non_null_values) and col in feature_ranges:
            min_val = feature_ranges[col]['min']
            max_val = feature_ranges[col]['max']
            range_violation_count = df[col].apply(lambda x: x < min_val or x > max_val if pd.notna(x) else False).sum()
            range_violation_ratio = range_violation_count / num_rows
            col_results['range_min_expected'] = min_val
            col_results['range_max_expected'] = max_val
            col_results['range_violation_count'] = range_violation_count
            col_results['range_violation_ratio'] = range_violation_ratio
            range_status = 'PASS'
            if range_violation_ratio > dq_thresholds['range_violation_ratio']['fail']:
                range_status = 'FAIL'
                if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
            elif range_violation_ratio > dq_thresholds['range_violation_ratio']['warn']:
                range_status = 'WARN'
                if overall_status == 'PASS': overall_status = 'WARN'
        col_results['range_violation_status'] = range_status

        # 5. Cardinality Check (for categorical-like columns)
        unique_count = df[col].nunique()
        col_results['unique_values_count'] = unique_count
        
        cardinality_status = 'N/A'
        if consistent_type in [str, object, int] or unique_count <= 50: # Heuristic for categorical-like features
            if unique_count == dq_thresholds['cardinality_unique_count_min']['fail']: # Only one unique value
                cardinality_status = 'FAIL'
                if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
            elif unique_count < dq_thresholds['cardinality_unique_count_min']['warn']: # Too few unique values
                cardinality_status = 'WARN'
                if overall_status == 'PASS': overall_status = 'WARN'
            elif unique_count / num_rows > dq_thresholds['cardinality_unique_count_max_ratio']['fail']: # Too many unique values (e.g., like an ID)
                cardinality_status = 'FAIL'
                if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
            elif unique_count / num_rows > dq_thresholds['cardinality_unique_count_max_ratio']['warn']:
                cardinality_status = 'WARN'
                if overall_status == 'PASS': overall_status = 'WARN'
            else:
                cardinality_status = 'PASS'
        col_results['cardinality_status'] = cardinality_status

        dq_results[col] = col_results

    dq_results['overall_dataset_quality_status'] = overall_status
    return dq_results

# --- Execution ---
data_quality_results = perform_data_quality_checks(primary_data, assessment_config)

print(f"Overall Data Quality Status: {data_quality_results['overall_dataset_quality_status']}\n")

# Display results in a human-readable table
table_headers = ["Feature", "Metric", "Value", "Status", "Threshold (Warn/Fail)"]
table_data = []

# Dataset overall duplicates
dup_res = data_quality_results['dataset_overall']
table_data.append([
    "Dataset (Overall)", "Duplicate Rows Ratio", 
    f"{dup_res['duplicate_rows_ratio']:.2%}", dup_res['duplicate_rows_status'], 
    f"{assessment_config['data_quality_thresholds']['duplicate_rows_ratio']['warn']:.2%}/"
    f"{assessment_config['data_quality_thresholds']['duplicate_rows_ratio']['fail']:.2%}"
])

for col, metrics in data_quality_results.items():
    if col in ['dataset_overall', 'overall_dataset_quality_status']:
        continue

    # Missingness
    table_data.append([
        col, "Missingness Ratio", 
        f"{metrics['missing_ratio']:.2%}", metrics['missing_status'], 
        f"{assessment_config['data_quality_thresholds']['missingness_ratio']['warn']:.2%}/"
        f"{assessment_config['data_quality_thresholds']['missingness_ratio']['fail']:.2%}"
    ])
    
    # Type Consistency
    table_data.append([
        col, "Type Inconsistency Ratio", 
        f"{metrics['type_inconsistency_ratio']:.2%} ({metrics['consistent_type']})", metrics['type_consistency_status'],
        f"{assessment_config['data_quality_thresholds']['type_inconsistency_ratio']['warn']:.2%}/"
        f"{assessment_config['data_quality_thresholds']['type_inconsistency_ratio']['fail']:.2%}"
    ])

    # Range Violations
    if metrics['range_violation_status'] != 'N/A':
        table_data.append([
            col, "Range Violation Ratio", 
            f"{metrics['range_violation_ratio']:.2%}", metrics['range_violation_status'], 
            f"{assessment_config['data_quality_thresholds']['range_violation_ratio']['warn']:.2%}/"
            f"{assessment_config['data_quality_thresholds']['range_violation_ratio']['fail']:.2%}"
        ])
    
    # Cardinality
    table_data.append([
        col, "Unique Values Count", 
        f"{metrics['unique_values_count']}", metrics['cardinality_status'], 
        f"< {assessment_config['data_quality_thresholds']['cardinality_unique_count_min']['warn']}"
        f" or > {assessment_config['data_quality_thresholds']['cardinality_unique_count_max_ratio']['warn']:.0%}"
        f" (W) / < {assessment_config['data_quality_thresholds']['cardinality_unique_count_min']['fail']}"
        f" or > {assessment_config['data_quality_thresholds']['cardinality_unique_count_max_ratio']['fail']:.0%}"
        f" (F)"
    ])

print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
```

### Explanation of Execution

Maya's data quality assessment reveals critical insights. The table clearly shows features flagged with 'WARN' or 'FAIL'. For example, `income` might have a 'FAIL' for missingness, indicating a significant portion of income data is absent. `credit_score` might show a 'FAIL' for type inconsistency due to non-numeric entries, which would break any numerical operations. `age` could have a 'WARN' for range violations, requiring a cleanup of outlier values.

These findings directly inform Maya's next steps:
-   **Missing Data:** Maya must decide on an imputation strategy (e.g., mean, median, predictive imputation) or whether the column should be dropped.
-   **Type Inconsistencies:** Requires data cleaning to convert values to the correct type or remove corrupted entries.
-   **Range Violations:** Outliers need to be handled, either corrected, capped, or removed, to prevent skewed model learning.

By addressing these issues proactively, Maya ensures the credit approval model receives reliable data, preventing it from making decisions based on faulty inputs.

---

## 3. Bias Metric Computation: Ensuring Fairness in Data

### Story + Context + Real-World Relevance

At FinTech Innovators Inc., ensuring fairness and avoiding discrimination in credit decisions is not just a regulatory requirement but a core ethical principle. Maya understands that biases present in the training data can be learned and amplified by models, leading to unfair outcomes for certain demographic groups. Before training the credit approval model, she must quantify any inherent biases within the new dataset. This assessment helps her identify if the raw data itself exhibits disparities in credit repayment outcomes across sensitive attributes like `marital_status` or `region`.

Since we are in a pre-training context (no model predictions yet), we will focus on measuring statistical parity and outcome disparities based on the *actual* target variable distributions across protected groups.

**Demographic Parity Difference (DPD):** Measures the difference in the proportion of the favorable outcome (e.g., loan repaid) between an unprivileged group and a privileged group. A value close to 0 indicates demographic parity.
$$ DPD = P(Y=1 | A_{unprivileged}) - P(Y=1 | A_{privileged}) $$
Where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{unprivileged}$ and $A_{privileged}$ representing the unprivileged and privileged groups, respectively.

**Disparate Impact Ratio (DIR):** Measures the ratio of the favorable outcome proportion for the unprivileged group to the privileged group. A value near 1 suggests no disparate impact. Values significantly below 1 (e.g., < 0.8) indicate the unprivileged group is less likely to receive the favorable outcome, while values significantly above 1 (e.g., > 1.25) indicate the unprivileged group is more likely.
$$ DIR = \frac{P(Y=1 | A_{unprivileged})}{P(Y=1 | A_{privileged})} $$

**Proxy True Positive Rate Gap (TPR Gap):** In a pre-training context, this can be interpreted as the difference in the *actual positive outcome rate* (prevalence of $Y=1$) between unprivileged and privileged groups. A value close to 0 indicates similar rates of positive outcomes for both groups in the raw data.
$$ \text{Proxy TPR Gap} = P(Y=1 | A_{unprivileged}) - P(Y=1 | A_{privileged}) $$

**Proxy False Positive Rate Gap (FPR Gap):** Similarly, in a pre-training context, this can be interpreted as the difference in the *actual negative outcome rate* (prevalence of $Y=0$) between unprivileged and privileged groups. A value close to 0 indicates similar rates of negative outcomes for both groups in the raw data.
$$ \text{Proxy FPR Gap} = P(Y=0 | A_{unprivileged}) - P(Y=0 | A_{privileged}) $$

```python
def compute_bias_metrics(df, target_column, sensitive_attributes, protected_groups_config, config):
    """
    Computes bias metrics for specified sensitive attributes.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        target_column (str): Name of the binary target label column (0 or 1).
        sensitive_attributes (list): List of column names considered sensitive.
        protected_groups_config (dict): Configuration mapping sensitive attributes to privileged/unprivileged groups.
        config (dict): Configuration dictionary containing bias thresholds.

    Returns:
        dict: A dictionary summarizing bias metrics per sensitive attribute.
    """
    bias_results = {}
    overall_status = 'PASS'
    bias_thresholds = config['bias_thresholds']

    for attr in sensitive_attributes:
        if attr not in df.columns:
            print(f"Warning: Sensitive attribute '{attr}' not found in DataFrame. Skipping bias check.")
            continue
        if attr not in protected_groups_config:
            print(f"Warning: Protected group definition for '{attr}' not found. Skipping bias check.")
            continue

        attr_results = {}
        privileged_group = protected_groups_config[attr]['privileged']
        unprivileged_groups = protected_groups_config[attr]['unprivileged']
        
        # Ensure target column is numeric (0 or 1)
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"Target column '{target_column}' must be numeric (0 or 1) for bias metrics.")
        
        # Calculate P(Y=1 | A=group) for all groups
        outcome_rates = df.groupby(attr)[target_column].mean() # P(Y=1 | A=group)
        
        # Determine the rate for the privileged group
        p_privileged = outcome_rates.get(privileged_group, 0)
        
        # Determine the rate for the unprivileged groups (average of specified unprivileged)
        p_unprivileged_list = [outcome_rates.get(group, 0) for group in unprivileged_groups if group in outcome_rates.index]
        p_unprivileged = np.mean(p_unprivileged_list) if p_unprivileged_list else 0

        # Demographic Parity Difference (DPD)
        dpd = p_unprivileged - p_privileged
        dpd_status = 'PASS'
        if abs(dpd) > bias_thresholds['demographic_parity_difference']['fail']:
            dpd_status = 'FAIL'
            if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
        elif abs(dpd) > bias_thresholds['demographic_parity_difference']['warn']:
            dpd_status = 'WARN'
            if overall_status == 'PASS': overall_status = 'WARN'
        attr_results['demographic_parity_difference'] = dpd
        attr_results['demographic_parity_difference_status'] = dpd_status

        # Disparate Impact Ratio (DIR)
        dir_val = p_unprivileged / p_privileged if p_privileged != 0 else np.inf
        dir_status = 'PASS'
        if dir_val < bias_thresholds['disparate_impact_ratio']['fail_lower'] or \
           dir_val > bias_thresholds['disparate_impact_ratio']['fail_upper']:
            dir_status = 'FAIL'
            if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
        elif dir_val < bias_thresholds['disparate_impact_ratio']['warn_lower'] or \
             dir_val > bias_thresholds['disparate_impact_ratio']['warn_upper']:
            dir_status = 'WARN'
            if overall_status == 'PASS': overall_status = 'WARN'
        attr_results['disparate_impact_ratio'] = dir_val
        attr_results['disparate_impact_ratio_status'] = dir_status

        # Proxy TPR Gap (difference in actual positive outcome rates)
        proxy_tpr_gap = p_unprivileged - p_privileged
        proxy_tpr_gap_status = 'PASS'
        if abs(proxy_tpr_gap) > bias_thresholds['proxy_tpr_gap']['fail']:
            proxy_tpr_gap_status = 'FAIL'
            if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
        elif abs(proxy_tpr_gap) > bias_thresholds['proxy_tpr_gap']['warn']:
            proxy_tpr_gap_status = 'WARN'
            if overall_status == 'PASS': overall_status = 'WARN'
        attr_results['proxy_tpr_gap'] = proxy_tpr_gap
        attr_results['proxy_tpr_gap_status'] = proxy_tpr_gap_status

        # Proxy FPR Gap (difference in actual negative outcome rates)
        # P(Y=0 | A=group) = 1 - P(Y=1 | A=group)
        p_negative_unprivileged = 1 - p_unprivileged
        p_negative_privileged = 1 - p_privileged
        proxy_fpr_gap = p_negative_unprivileged - p_negative_privileged
        proxy_fpr_gap_status = 'PASS'
        if abs(proxy_fpr_gap) > bias_thresholds['proxy_fpr_gap']['fail']:
            proxy_fpr_gap_status = 'FAIL'
            if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
        elif abs(proxy_fpr_gap) > bias_thresholds['proxy_fpr_gap']['warn']:
            proxy_fpr_gap_status = 'WARN'
            if overall_status == 'PASS': overall_status = 'WARN'
        attr_results['proxy_fpr_gap'] = proxy_fpr_gap
        attr_results['proxy_fpr_gap_status'] = proxy_fpr_gap_status

        bias_results[attr] = {
            'privileged_group': privileged_group,
            'unprivileged_groups': unprivileged_groups,
            'p_privileged_outcome': p_privileged,
            'p_unprivileged_outcome': p_unprivileged,
            **attr_results
        }
    
    bias_results['overall_bias_status'] = overall_status
    return bias_results

# --- Execution ---
bias_metrics_results = compute_bias_metrics(
    primary_data, 
    assessment_config['target_column'], 
    assessment_config['sensitive_attributes'], 
    assessment_config['protected_groups'], 
    assessment_config
)

print(f"Overall Bias Status: {bias_metrics_results['overall_bias_status']}\n")

# Display results in a human-readable table
table_headers = ["Sensitive Attribute", "Metric", "Value", "Status", "Threshold (Warn/Fail)"]
table_data = []

for attr, metrics in bias_metrics_results.items():
    if attr == 'overall_bias_status':
        continue
    
    warn_dpd = assessment_config['bias_thresholds']['demographic_parity_difference']['warn']
    fail_dpd = assessment_config['bias_thresholds']['demographic_parity_difference']['fail']
    table_data.append([
        attr, "Demographic Parity Difference", 
        f"{metrics['demographic_parity_difference']:.4f}", metrics['demographic_parity_difference_status'],
        f"Abs > {warn_dpd:.2f} (W) / > {fail_dpd:.2f} (F)"
    ])

    warn_dir_l = assessment_config['bias_thresholds']['disparate_impact_ratio']['warn_lower']
    warn_dir_u = assessment_config['bias_thresholds']['disparate_impact_ratio']['warn_upper']
    fail_dir_l = assessment_config['bias_thresholds']['disparate_impact_ratio']['fail_lower']
    fail_dir_u = assessment_config['bias_thresholds']['disparate_impact_ratio']['fail_upper']
    table_data.append([
        attr, "Disparate Impact Ratio", 
        f"{metrics['disparate_impact_ratio']:.4f}", metrics['disparate_impact_ratio_status'],
        f"< {warn_dir_l:.2f} or > {warn_dir_u:.2f} (W) / < {fail_dir_l:.2f} or > {fail_dir_u:.2f} (F)"
    ])

    warn_tpr_gap = assessment_config['bias_thresholds']['proxy_tpr_gap']['warn']
    fail_tpr_gap = assessment_config['bias_thresholds']['proxy_tpr_gap']['fail']
    table_data.append([
        attr, "Proxy TPR Gap", 
        f"{metrics['proxy_tpr_gap']:.4f}", metrics['proxy_tpr_gap_status'],
        f"Abs > {warn_tpr_gap:.2f} (W) / > {fail_tpr_gap:.2f} (F)"
    ])

    warn_fpr_gap = assessment_config['bias_thresholds']['proxy_fpr_gap']['warn']
    fail_fpr_gap = assessment_config['bias_thresholds']['proxy_fpr_gap']['fail']
    table_data.append([
        attr, "Proxy FPR Gap", 
        f"{metrics['proxy_fpr_gap']:.4f}", metrics['proxy_fpr_gap_status'],
        f"Abs > {warn_fpr_gap:.2f} (W) / > {fail_fpr_gap:.2f} (F)"
    ])

print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
```

### Explanation of Execution

The bias metrics report provides Maya with quantitative evidence of fairness (or lack thereof) in the raw data. For instance, if the `region` attribute shows a Disparate Impact Ratio significantly below 1 (e.g., for the 'South' region), it indicates that customers from this region are less likely to have favorable credit outcomes ($Y=1$) in the dataset compared to the privileged 'North' region. This is a critical 'WARN' or 'FAIL' condition for FinTech Innovators Inc.

Maya must now consider strategies to mitigate these biases *before* model training. This could involve:
-   **Data collection review:** Investigating if the data collection process itself introduced biases.
-   **Feature engineering:** Creating new features that might reduce reliance on sensitive attributes.
-   **Resampling techniques:** Over-sampling underrepresented groups or under-sampling overrepresented groups to balance the dataset.

These steps are crucial for FinTech Innovators Inc. to ensure equitable credit access and avoid legal or reputational risks.

---

## 4. Drift Detection: Monitoring Data Distribution Shifts

### Story + Context + Real-World Relevance

FinTech Innovators Inc. operates in a dynamic financial market, where customer behaviors and economic conditions can change rapidly. Maya understands that a credit approval model trained on old data might become less accurate if the underlying data distribution shifts over time. This phenomenon, known as "data drift," can severely degrade model performance in production. To mitigate this risk, Maya needs to compare the new credit application dataset against a historical "baseline" dataset (the data the current production model was originally trained on).

The **Population Stability Index (PSI)** is a widely used metric to quantify data drift for numerical features. It measures how much a variable's distribution has changed from a baseline period to a current period.

**Population Stability Index (PSI):** For each feature, the PSI is calculated by:
1.  Dividing the range of the feature into several bins (e.g., 10 bins).
2.  Calculating the percentage of observations in each bin for both the baseline ($P_{0i}$) and the current ($P_i$) datasets.
3.  Summing the contributions from each bin:
    $$ \text{PSI} = \sum_{i=1}^{N} (P_i - P_{0i}) \times \ln\left(\frac{P_i}{P_{0i}}\right) $$
    Where $N$ is the number of bins, $P_i$ is the percentage of observations in bin $i$ for the current dataset, and $P_{0i}$ is the percentage of observations in bin $i$ for the baseline dataset. A small constant (epsilon) is typically added to $P_i$ and $P_{0i}$ to avoid issues with zero values in the logarithm.

Common interpretations for PSI:
-   PSI < 0.1: No significant shift (PASS)
-   0.1 <= PSI < 0.25: Small shift (WARN)
-   PSI >= 0.25: Significant shift (FAIL)

```python
def calculate_psi(df_primary, df_baseline, numeric_cols, config, num_bins=10, epsilon=1e-6):
    """
    Calculates Population Stability Index (PSI) for numerical features.

    Args:
        df_primary (pd.DataFrame): The primary (new) dataset.
        df_baseline (pd.DataFrame): The baseline (historical) dataset.
        numeric_cols (list): List of numerical column names to calculate PSI for.
        config (dict): Configuration dictionary containing PSI thresholds.
        num_bins (int): Number of bins to use for distribution comparison.
        epsilon (float): Small constant to avoid log(0) issues.

    Returns:
        dict: A dictionary summarizing PSI results per numerical feature.
    """
    psi_results = {}
    overall_status = 'PASS'
    psi_thresholds = config['drift_thresholds']['psi']

    for col in numeric_cols:
        if col not in df_primary.columns or col not in df_baseline.columns:
            print(f"Warning: Column '{col}' not found in both datasets. Skipping PSI.")
            continue
        
        # Ensure the column is numeric in both dataframes
        if not pd.api.types.is_numeric_dtype(df_primary[col]) or not pd.api.types.is_numeric_dtype(df_baseline[col]):
            print(f"Warning: Column '{col}' is not numeric. Skipping PSI calculation.")
            continue

        # Combine data to determine consistent bin edges across both datasets
        all_data = pd.concat([df_primary[col].dropna(), df_baseline[col].dropna()])
        if all_data.empty:
            psi_results[col] = {'psi': np.nan, 'status': 'N/A', 'error': 'No data for PSI calculation'}
            continue

        min_val = all_data.min()
        max_val = all_data.max()
        if min_val == max_val: # Handle constant features
             psi_results[col] = {'psi': 0.0, 'status': 'PASS', 'error': 'Constant feature'}
             continue

        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Handle cases where bins are not unique (e.g., if data range is very small)
        if len(np.unique(bins)) < num_bins + 1:
            # Fallback to pandas qcut or cut with fewer bins if possible
            try:
                # Use qcut for quantile-based bins if issues with linspace
                primary_bins = pd.qcut(df_primary[col].dropna(), q=num_bins, labels=False, duplicates='drop')
                baseline_bins = pd.qcut(df_baseline[col].dropna(), q=num_bins, labels=False, duplicates='drop')
                bin_edges = pd.qcut(all_data, q=num_bins, retbins=True, duplicates='drop')[1]
                bins = bin_edges
            except Exception:
                # If qcut also fails for some reason (e.g., very few unique values), use simple cut
                primary_bins = pd.cut(df_primary[col].dropna(), bins=num_bins, labels=False, include_lowest=True)
                baseline_bins = pd.cut(df_baseline[col].dropna(), bins=num_bins, labels=False, include_lowest=True)
                bin_edges = pd.cut(all_data, bins=num_bins, retbins=True, include_lowest=True)[1]
                bins = bin_edges
            
            # Re-evaluate number of bins after dynamic adjustment
            actual_num_bins = len(bins) - 1
            if actual_num_bins < 1:
                 psi_results[col] = {'psi': 0.0, 'status': 'PASS', 'error': 'Not enough distinct values for binning'}
                 continue
        else:
            primary_bins = pd.cut(df_primary[col].dropna(), bins=bins, labels=False, include_lowest=True)
            baseline_bins = pd.cut(df_baseline[col].dropna(), bins=bins, labels=False, include_lowest=True)

        # Calculate population percentages for each bin
        primary_counts = primary_bins.value_counts(normalize=True).sort_index()
        baseline_counts = baseline_bins.value_counts(normalize=True).sort_index()

        # Align indices (bins) and fill missing with 0 for bins present in one but not the other
        all_bin_indices = sorted(list(set(primary_counts.index).union(set(baseline_counts.index))))
        p_primary = primary_counts.reindex(all_bin_indices, fill_value=0)
        p_baseline = baseline_counts.reindex(all_bin_indices, fill_value=0)
        
        # Ensure no zero values for log calculation
        p_primary = p_primary + epsilon
        p_baseline = p_baseline + epsilon

        # Calculate PSI
        psi = np.sum((p_primary - p_baseline) * np.log(p_primary / p_baseline))
        
        psi_status = 'PASS'
        if psi > psi_thresholds['fail']:
            psi_status = 'FAIL'
            if overall_status == 'PASS' or overall_status == 'WARN': overall_status = 'FAIL'
        elif psi > psi_thresholds['warn']:
            psi_status = 'WARN'
            if overall_status == 'PASS': overall_status = 'WARN'
        
        psi_results[col] = {'psi': psi, 'status': psi_status}

    psi_results['overall_drift_status'] = overall_status
    return psi_results

# --- Execution ---
if baseline_data is not None:
    # Identify numeric columns for PSI calculation
    numeric_columns = primary_data.select_dtypes(include=np.number).columns.tolist()
    # Exclude the target column if it's binary and not meant for continuous drift (unless specified)
    if assessment_config['target_column'] in numeric_columns:
        numeric_columns.remove(assessment_config['target_column']) 
    
    drift_detection_results = calculate_psi(primary_data, baseline_data, numeric_columns, assessment_config)
    print(f"Overall Drift Status: {drift_detection_results['overall_drift_status']}\n")

    # Display results in a human-readable table
    table_headers = ["Feature", "PSI Value", "Status", "Threshold (Warn/Fail)"]
    table_data = []

    for col, metrics in drift_detection_results.items():
        if col == 'overall_drift_status':
            continue
        
        warn_psi = assessment_config['drift_thresholds']['psi']['warn']
        fail_psi = assessment_config['drift_thresholds']['psi']['fail']

        psi_val_str = f"{metrics['psi']:.4f}" if not np.isnan(metrics['psi']) else metrics.get('error', 'N/A')
        
        table_data.append([
            col, psi_val_str, metrics['status'],
            f"> {warn_psi:.2f} (W) / > {fail_psi:.2f} (F)"
        ])

    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    
    # Optional: Visualize drift for a few key features
    print("\nVisualizing drift for 'income' and 'credit_score':")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(primary_data['income'].dropna(), kde=True, color='skyblue', label='Primary', ax=axes[0], stat='density', alpha=0.7)
    sns.histplot(baseline_data['income'].dropna(), kde=True, color='salmon', label='Baseline', ax=axes[0], stat='density', alpha=0.7)
    axes[0].set_title(f'Income Distribution (PSI: {drift_detection_results.get("income", {}).get("psi", np.nan):.2f})')
    axes[0].legend()
    
    sns.histplot(primary_data['credit_score'].dropna(), kde=True, color='skyblue', label='Primary', ax=axes[1], stat='density', alpha=0.7)
    sns.histplot(baseline_data['credit_score'].dropna(), kde=True, color='salmon', label='Baseline', ax=axes[1], stat='density', alpha=0.7)
    axes[1].set_title(f'Credit Score Distribution (PSI: {drift_detection_results.get("credit_score", {}).get("psi", np.nan):.2f})')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

else:
    drift_detection_results = {'overall_drift_status': 'N/A', 'message': 'Baseline dataset not provided for drift detection.'}
    print(drift_detection_results['message'])
```

### Explanation of Execution

Maya examines the PSI results to detect significant shifts. For instance, if `income` or `credit_score` show a 'WARN' or 'FAIL' PSI value, it means the distribution of these critical features in the new dataset has changed notably from the baseline. The visualizations provide an intuitive understanding of these shifts. A shift in `income` distribution, for example, could indicate changing economic conditions or a different customer segment applying for credit.

This insight is crucial for Maya:
-   **Model Re-training:** A significant drift (`FAIL`) in key features might necessitate immediate re-training of the credit approval model using the new data.
-   **Monitoring Strategy:** Even a 'WARN' suggests closer monitoring of the model's performance on these features in production.
-   **Business Context:** It prompts FinTech Innovators Inc. to investigate the business reasons behind the drift, informing both model development and business strategy.

By systematically identifying data drift, Maya helps FinTech Innovators Inc. maintain accurate and relevant models, preventing unexpected performance degradation in live environments.

---

## 5. Overall Readiness Decision and Comprehensive Reporting

### Story + Context + Real-World Relevance

After conducting thorough data quality, bias, and drift assessments, Maya needs to synthesize all findings into a clear, deterministic decision regarding the dataset's readiness for model training. This final decision, along with a comprehensive report, is critical for communication with Model Validators (Persona 2) and Risk/Compliance Partners (Persona 3) at FinTech Innovators Inc. It provides a transparent, quantitative basis for proceeding or halting model development, ensuring accountability and adherence to governance standards.

The overall readiness decision follows a strict logic:
-   **DO NOT DEPLOY:** If any 'FAIL' condition is identified across data quality, bias, or drift metrics. This indicates severe issues that must be addressed before proceeding.
-   **PROCEED WITH MITIGATION:** If there are only 'WARN' conditions, but no 'FAIL's. This means the dataset has minor issues that can likely be mitigated (e.g., specific data cleaning, bias-aware modeling) while proceeding with development.
-   **PROCEED:** If all checks result in a 'PASS'. This indicates the dataset is robust and ready for model training without immediate, significant concerns.

```python
def make_readiness_decision(data_quality_results, bias_metrics_results, drift_detection_results):
    """
    Determines the overall dataset readiness based on individual assessment results.

    Args:
        data_quality_results (dict): Results from data quality checks.
        bias_metrics_results (dict): Results from bias metrics computation.
        drift_detection_results (dict): Results from drift detection.

    Returns:
        str: The overall readiness decision ('DO NOT DEPLOY', 'PROCEED WITH MITIGATION', 'PROCEED').
    """
    overall_status = 'PROCEED' # Start with the most optimistic, downgrade as issues are found

    # Check Data Quality
    if data_quality_results['overall_dataset_quality_status'] == 'FAIL':
        overall_status = 'DO NOT DEPLOY'
    elif data_quality_results['overall_dataset_quality_status'] == 'WARN' and overall_status == 'PROCEED':
        overall_status = 'PROCEED WITH MITIGATION'

    # Check Bias Metrics
    if bias_metrics_results['overall_bias_status'] == 'FAIL':
        overall_status = 'DO NOT DEPLOY'
    elif bias_metrics_results['overall_bias_status'] == 'WARN' and overall_status == 'PROCEED':
        overall_status = 'PROCEED WITH MITIGATION'

    # Check Drift Detection (only if a baseline was provided and analysis performed)
    if 'overall_drift_status' in drift_detection_results and drift_detection_results['overall_drift_status'] != 'N/A':
        if drift_detection_results['overall_drift_status'] == 'FAIL':
            overall_status = 'DO NOT DEPLOY'
        elif drift_detection_results['overall_drift_status'] == 'WARN' and overall_status == 'PROCEED':
            overall_status = 'PROCEED WITH MITIGATION'
    
    return overall_status

def generate_reports(data_quality_results, bias_metrics_results, drift_detection_results, readiness_decision, config, primary_data_path, baseline_data_path=None, output_dir='reports'):
    """
    Generates JSON and Markdown reports summarizing all findings.

    Args:
        data_quality_results (dict): Results from data quality checks.
        bias_metrics_results (dict): Results from bias metrics computation.
        drift_detection_results (dict): Results from drift detection.
        readiness_decision (str): The overall readiness decision.
        config (dict): The configuration used for the assessment.
        primary_data_path (str): Path to the primary dataset.
        baseline_data_path (str, optional): Path to the baseline dataset.
        output_dir (str): Directory to save the reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_folder = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(report_folder, exist_ok=True)

    # 1. data_quality_report.json
    with open(os.path.join(report_folder, 'data_quality_report.json'), 'w') as f:
        json.dump(data_quality_results, f, indent=4)

    # 2. bias_metrics.json
    with open(os.path.join(report_folder, 'bias_metrics.json'), 'w') as f:
        json.dump(bias_metrics_results, f, indent=4)

    # 3. drift_report.json
    with open(os.path.join(report_folder, 'drift_report.json'), 'w') as f:
        json.dump(drift_detection_results, f, indent=4)

    # 4. config_snapshot.json
    config_snapshot = {
        'primary_data_path': primary_data_path,
        'baseline_data_path': baseline_data_path,
        **config
    }
    with open(os.path.join(report_folder, 'config_snapshot.json'), 'w') as f:
        json.dump(config_snapshot, f, indent=4)

    # 5. session04_executive_summary.md
    summary_path = os.path.join(report_folder, 'session04_executive_summary.md')
    with open(summary_path, 'w') as f:
        f.write(f"# Data Readiness Assessment Summary (Run ID: {run_id})\n\n")
        f.write(f"**Date & Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Primary Dataset:** `{primary_data_path}`\n")
        if baseline_data_path:
            f.write(f"**Baseline Dataset:** `{baseline_data_path}`\n")
        f.write(f"**Target Column:** `{config['target_column']}`\n")
        f.write(f"**Sensitive Attributes:** {', '.join(config['sensitive_attributes'])}\n\n")
        f.write(f"## Overall Dataset Readiness Decision: **{readiness_decision}**\n\n")
        f.write("---\n\n")

        f.write("## 1. Data Quality Assessment\n")
        f.write(f"**Overall Status:** `{data_quality_results['overall_dataset_quality_status']}`\n\n")
        f.write("### Key Findings:\n")
        for col, metrics in data_quality_results.items():
            if col in ['dataset_overall', 'overall_dataset_quality_status']:
                continue
            if metrics['missing_status'] != 'PASS' or \
               metrics['type_consistency_status'] != 'PASS' or \
               metrics['range_violation_status'] not in ['PASS', 'N/A'] or \
               metrics['cardinality_status'] not in ['PASS', 'N/A']:
                f.write(f"- **Feature `{col}`:**\n")
                if metrics['missing_status'] != 'PASS':
                    f.write(f"  - Missingness: {metrics['missing_ratio']:.2%} (`{metrics['missing_status']}`)\n")
                if metrics['type_consistency_status'] != 'PASS':
                    f.write(f"  - Type Inconsistency: {metrics['type_inconsistency_ratio']:.2%} (`{metrics['type_consistency_status']}`)\n")
                if metrics['range_violation_status'] not in ['PASS', 'N/A']:
                    f.write(f"  - Range Violations: {metrics['range_violation_ratio']:.2%} (`{metrics['range_violation_status']}`)\n")
                if metrics['cardinality_status'] not in ['PASS', 'N/A']:
                    f.write(f"  - Cardinality: {metrics['unique_values_count']} unique values (`{metrics['cardinality_status']}`)\n")
        if data_quality_results['dataset_overall']['duplicate_rows_status'] != 'PASS':
             f.write(f"- **Dataset Overall:** Duplicate Rows: {data_quality_results['dataset_overall']['duplicate_rows_ratio']:.2%} (`{data_quality_results['dataset_overall']['duplicate_rows_status']}`)\n")

        f.write("\n---\n\n")

        f.write("## 2. Bias Metrics Assessment\n")
        f.write(f"**Overall Status:** `{bias_metrics_results['overall_bias_status']}`\n\n")
        f.write("### Key Findings:\n")
        for attr, metrics in bias_metrics_results.items():
            if attr == 'overall_bias_status':
                continue
            if metrics['demographic_parity_difference_status'] != 'PASS' or \
               metrics['disparate_impact_ratio_status'] != 'PASS' or \
               metrics['proxy_tpr_gap_status'] != 'PASS' or \
               metrics['proxy_fpr_gap_status'] != 'PASS':
                f.write(f"- **Sensitive Attribute `{attr}`:** (Privileged: `{metrics['privileged_group']}`, Unprivileged: `{', '.join(metrics['unprivileged_groups'])}`)\n")
                if metrics['demographic_parity_difference_status'] != 'PASS':
                    f.write(f"  - Demographic Parity Difference: {metrics['demographic_parity_difference']:.4f} (`{metrics['demographic_parity_difference_status']}`)\n")
                if metrics['disparate_impact_ratio_status'] != 'PASS':
                    f.write(f"  - Disparate Impact Ratio: {metrics['disparate_impact_ratio']:.4f} (`{metrics['disparate_impact_ratio_status']}`)\n")
                if metrics['proxy_tpr_gap_status'] != 'PASS':
                    f.write(f"  - Proxy TPR Gap: {metrics['proxy_tpr_gap']:.4f} (`{metrics['proxy_tpr_gap_status']}`)\n")
                if metrics['proxy_fpr_gap_status'] != 'PASS':
                    f.write(f"  - Proxy FPR Gap: {metrics['proxy_fpr_gap']:.4f} (`{metrics['proxy_fpr_gap_status']}`)\n")
        f.write("\n---\n\n")

        f.write("## 3. Drift Detection (Population Stability Index - PSI)\n")
        if 'overall_drift_status' in drift_detection_results and drift_detection_results['overall_drift_status'] != 'N/A':
            f.write(f"**Overall Status:** `{drift_detection_results['overall_drift_status']}`\n\n")
            f.write("### Key Findings:\n")
            for col, metrics in drift_detection_results.items():
                if col == 'overall_drift_status':
                    continue
                if metrics['status'] != 'PASS' and metrics['status'] != 'N/A':
                    f.write(f"- **Feature `{col}`:** PSI = {metrics['psi']:.4f} (`{metrics['status']}`)\n")
        else:
            f.write("Baseline dataset not provided for drift detection. No drift assessment performed.\n")

        f.write("\n---\n\n")
        f.write("## Recommendations for FinTech Innovators Inc.:\n")
        if readiness_decision == 'DO NOT DEPLOY':
            f.write("- **Immediate Action Required:** Dataset contains critical 'FAIL' conditions. Do NOT proceed with model training or deployment until all identified issues (e.g., high missingness, type inconsistencies, severe bias, significant drift) are thoroughly addressed and re-assessed.\n")
        elif readiness_decision == 'PROCEED WITH MITIGATION':
            f.write("- **Proceed with Caution:** Dataset contains 'WARN' conditions. It is recommended to proceed with model training, but implement specific mitigation strategies for the identified data quality, bias, or drift issues. This could involve targeted data cleaning, fairness-aware pre-processing, or careful monitoring in production.\n")
        else: # PROCEED
            f.write("- **Proceed with Confidence:** Dataset meets all defined quality standards. You can proceed with model training and validation, while maintaining standard monitoring practices.\n")

    return report_folder

# --- Execution ---
readiness_decision = make_readiness_decision(data_quality_results, bias_metrics_results, drift_detection_results)
reports_folder_path = generate_reports(
    data_quality_results, 
    bias_metrics_results, 
    drift_detection_results, 
    readiness_decision, 
    assessment_config, 
    primary_data_path, 
    baseline_data_path
)

print(f"Overall Dataset Readiness Decision: **{readiness_decision}**")
print(f"Reports generated and saved to: `{reports_folder_path}`")

# Display the executive summary for immediate review
print("\n--- Executive Summary ---")
with open(os.path.join(reports_folder_path, 'session04_executive_summary.md'), 'r') as f:
    print(f.read())
```

### Explanation of Execution

Maya has now generated a comprehensive suite of reports. The `overall_dataset_readiness_status` provides a definitive answer for her stakeholders. For FinTech Innovators Inc., a 'DO NOT DEPLOY' status (due to the simulated 'FAIL' conditions in our sample data) means Maya must halt the model development pipeline. She then shares the `session04_executive_summary.md` with the Model Validator and Risk/Compliance Partner. This markdown document provides a high-level overview, highlighting critical issues (e.g., "Feature `income`: Missingness: 25.00% (`FAIL`)", "Sensitive Attribute `region`: Disparate Impact Ratio: 0.6700 (`FAIL`)"), and the clear recommendation.

The JSON reports (e.g., `data_quality_report.json`) provide the granular details necessary for deeper investigation by the technical team. This structured reporting ensures that FinTech Innovators Inc. maintains a clear audit trail for all data-related decisions and adheres to its internal governance framework.

---

## 6. Exporting Audit Artifacts: Ensuring Traceability and Compliance

### Story + Context + Real-World Relevance

For FinTech Innovators Inc., regulatory compliance and internal auditability are paramount. Every decision related to model development, especially data quality, must be fully traceable and defensible. Maya's final, critical step is to bundle all generated reports, configurations, and an evidence manifest (containing SHA-256 hashes of all artifacts) into a secure, version-controlled zip archive. This ensures that the assessment results are immutable, tamper-evident, and readily available for future audits or reviews by risk and compliance teams.

**Evidence Manifest:** A record of all generated files, including their SHA-256 hash. The SHA-256 hash is a cryptographic checksum that uniquely identifies the content of a file. If even a single bit in the file changes, its SHA-256 hash will be drastically different.
$$ \text{SHA-256 Hash} = \text{SHA256}(\text{file\_content}) $$
This cryptographic integrity check confirms that the reports have not been altered since their generation.

```python
def create_evidence_manifest(report_folder):
    """
    Creates an evidence manifest (SHA-256 hashes) for all files in the report folder.

    Args:
        report_folder (str): The directory containing the generated reports.

    Returns:
        dict: A dictionary mapping file paths (relative to report_folder) to their SHA-256 hashes.
    """
    manifest = {}
    for root, _, files in os.walk(report_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, report_folder)
            
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            manifest[relative_path] = file_hash
    return manifest

def bundle_artifacts(report_folder, run_id, output_zip_dir='reports/archive'):
    """
    Bundles all generated reports and the evidence manifest into a zip archive.

    Args:
        report_folder (str): The directory containing the generated reports.
        run_id (str): Unique identifier for this run.
        output_zip_dir (str): Directory where the zip file will be saved.

    Returns:
        str: Path to the generated zip archive.
    """
    os.makedirs(output_zip_dir, exist_ok=True)
    zip_file_name = f"Session_04_{run_id}.zip"
    zip_file_path = os.path.join(output_zip_dir, zip_file_name)

    # Create the evidence manifest
    evidence_manifest = create_evidence_manifest(report_folder)
    manifest_path = os.path.join(report_folder, 'evidence_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(evidence_manifest, f, indent=4)
    
    # Add the manifest to the bundle for hashing itself
    manifest_relative_path = os.path.relpath(manifest_path, report_folder)
    with open(manifest_path, 'rb') as f:
        manifest_hash = hashlib.sha256(f.read()).hexdigest()
    evidence_manifest[manifest_relative_path] = manifest_hash # Update manifest with its own hash

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(report_folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # Ensure the path inside the zip is relative to the run_id folder
                arcname = os.path.relpath(file_path, os.path.dirname(report_folder))
                zf.write(file_path, arcname)
    
    print(f"Evidence manifest generated: `{manifest_path}`")
    return zip_file_path

# --- Execution ---
# Extract run_id from the reports_folder_path
run_id = os.path.basename(reports_folder_path).replace('run_', '')

zip_archive_path = bundle_artifacts(reports_folder_path, run_id)

print(f"All audit artifacts bundled into: `{zip_archive_path}`")

# Optional: Verify a hash from the manifest
manifest_content = {}
with open(os.path.join(reports_folder_path, 'evidence_manifest.json'), 'r') as f:
    manifest_content = json.load(f)

print("\n--- Verifying an Artifact Hash ---")
sample_file_to_verify = 'data_quality_report.json'
if sample_file_to_verify in manifest_content:
    original_hash = manifest_content[sample_file_to_verify]
    
    current_file_path = os.path.join(reports_folder_path, sample_file_to_verify)
    with open(current_file_path, 'rb') as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()
    
    print(f"File: {sample_file_to_verify}")
    print(f"Original Hash (from manifest): {original_hash}")
    print(f"Current Hash: {current_hash}")
    if original_hash == current_hash:
        print("Verification: PASS - Hashes match, file integrity confirmed.")
    else:
        print("Verification: FAIL - Hashes do NOT match, file may have been altered!")
else:
    print(f"Error: {sample_file_to_verify} not found in manifest.")
```

### Explanation of Execution

Maya has successfully created a zip archive containing all the assessment reports, the configuration snapshot, and a crucial `evidence_manifest.json` file. This manifest acts as a digital fingerprint for all generated documents. The verification step demonstrates how a Model Validator or Risk Partner at FinTech Innovators Inc. can confirm the integrity of any report by recomputing its hash and comparing it against the manifest. If any report were to be tampered with, the hash verification would fail, immediately flagging a potential issue.

This robust export mechanism ensures that FinTech Innovators Inc. can confidently demonstrate compliance, provide auditable evidence of data quality and fairness assessments, and maintain strict governance over its ML models throughout their lifecycle. Maya's work is complete, providing FinTech Innovators Inc. with the objective evidence needed to make informed decisions about its credit approval model.
