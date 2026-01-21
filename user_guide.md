id: 6970ede2f1dd2fc1624a17fb_user_guide
summary: Lab 4: Data Quality, Provenance & Bias Metrics Dashboard User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Navigating Data Quality, Provenance & Bias for ML Models

## 1. Setting the Stage: Data Loading and Configuration
Duration: 00:10:00

Welcome to this hands-on lab designed for ML Engineers, Model Validators, and Risk/Compliance Partners. In this codelab, we will step into the shoes of Maya, a Senior ML Engineer at "FinTech Innovators Inc.". Maya's team is responsible for developing and maintaining robust machine learning models that adhere to strict regulatory standards and deliver fair outcomes to customers.

Today, Maya is tasked with preparing a new dataset for a credit approval model. Before FinTech Innovators Inc. commits to costly model training and potential deployment, Maya needs to perform a comprehensive data quality and risk assessment. Her goal is to ensure the raw data meets fundamental quality standards, does not contain hidden biases, and has not drifted significantly from historical data, thereby preventing unnecessary model risk and ensuring compliance.

This initial section guides you, as Maya, through loading the new credit application dataset and configuring parameters for a tailored assessment. Defining these parameters upfront is crucial to align the assessment with FinTech Innovators Inc.'s specific model requirements and compliance policies.

<aside class="positive">
<b>Understanding the 'Why':</b> Poor data quality or undetected biases can lead to inaccurate model predictions, unfair outcomes, and significant financial and reputational risks for a company like FinTech Innovators Inc. This first step is the foundation for a trustworthy and compliant ML system.
</aside>

### Upload Your Datasets

First, you need to provide the data that will be assessed.
*   **Primary Dataset:** This is the new dataset you want to assess for your credit approval model.
*   **Optional Baseline Dataset:** This is a historical dataset, typically the one your current production model was trained on. It's used to detect if your new primary dataset has "drifted" or significantly changed over time.

1.  Locate the "Upload Datasets" section.
2.  Click "Upload Primary Dataset (CSV)" and select your primary data file.
3.  (Optional) Click "Upload Optional Baseline Dataset (CSV)" and select your baseline data file.

Once uploaded, you'll see a preview of your datasets, helping you verify that the correct data has been loaded.

### Configure Assessment Parameters

After uploading, you'll define the key elements of your assessment:

1.  **Select Target Label Column:** This is the column in your dataset that represents the outcome your model will predict (e.g., 'loan_approved', 'repaid'). Select the relevant column from the dropdown.
2.  **Select Sensitive Attributes:** These are columns that represent protected demographic characteristics (e.g., 'age', 'gender', 'marital_status', 'region'). Models must not discriminate based on these attributes. Select all relevant columns from the multiselect dropdown.

    <aside class="positive">
    For FinTech Innovators Inc., a key concern is ensuring fair lending practices. Therefore, identifying sensitive attributes is paramount to later check for potential biases.
    </aside>

3.  **Define Protected Groups:** Expand the "Define Protected Groups (e.g., Privileged/Unprivileged)" section. For each sensitive attribute you selected, you will define which values constitute the 'privileged' group and which constitute the 'unprivileged' group. For example, for 'marital_status', 'Married' might be privileged, and 'Single,Divorced' might be unprivileged. These definitions are crucial for bias metric computation.
4.  **Override Default Thresholds (Optional but Recommended):** Expand the "Override Default Thresholds" section. Maya can override predefined warning and failure thresholds for data quality, bias, and drift metrics to align with FinTech Innovators Inc.'s specific risk appetite.
    *   Review and adjust the thresholds for **Data Quality**, **Bias**, and **Drift (PSI)** metrics. These thresholds determine when an issue is flagged as a 'WARN' (minor concern) or 'FAIL' (major concern).
    *   For numerical features, you can also define **Feature Range Expectations** to identify values outside an acceptable business range.

### Apply Configuration

Once all parameters are set:

1.  Click the "Apply Configuration" button.
2.  The application will process your selections and display the "Current Assessment Configuration" in a JSON format.

<aside class="positive">
Maya has successfully loaded the datasets and reviewed the configuration. She can see the target column, the sensitive attributes, and the default (and any custom) thresholds for each metric. This initial setup is critical to ensure the assessment is aligned with project goals and regulatory requirements.
</aside>

## 2. Core Data Quality Assessment: Uncovering Raw Data Issues
Duration: 00:08:00

Before any sophisticated modeling, Maya must ensure the fundamental quality of the dataset. This means checking for common issues like missing values, duplicate entries, inconsistent data types, values outside expected ranges, and inappropriate cardinality for categorical features. Catching these problems early prevents downstream errors in model training, improves model robustness, and saves significant computational resources. For FinTech Innovators Inc., poor data quality could lead to inaccurate credit risk assessments, violating internal policies and potentially regulatory guidelines.

Maya will use the configured thresholds to assign a 'PASS', 'WARN', or 'FAIL' status to each quality aspect of every feature.

### Key Data Quality Metrics

*   **Missingness Ratio ($M_i$):** The proportion of missing values for feature $i$.
    $$ M_i = \frac{{\text{{Number of Missing Values in Feature }} i}}{{\text{{Total Number of Rows}}}} $$
    where $M_i$ is the missingness ratio for feature $i$, Number of Missing Values in Feature $i$ is the count of null values in column $i$, and Total Number of Rows is the total number of entries in the dataset.

*   **Duplicate Rows Ratio ($D$):** The proportion of rows that are exact duplicates of other rows in the dataset.
    $$ D = \frac{{\text{{Number of Duplicate Rows}}}}{{\text{{Total Number of Rows}}}} $$
    where $D$ is the duplicate rows ratio, Number of Duplicate Rows is the count of rows that are identical to another row, and Total Number of Rows is the total number of entries in the dataset.

*   **Type Inconsistency:** Measured as the ratio of non-conforming data types within a column. For example, if a numeric column contains string values, this metric will be high.

*   **Range Violation Ratio ($R_i$):** For numerical features, this is the ratio of values falling outside a predefined acceptable range.
    $$ R_i = \frac{{\text{{Number of Values Outside Expected Range for Feature }} i}}{{\text{{Total Number of Rows}}}} $$
    where $R_i$ is the range violation ratio for feature $i$, Number of Values Outside Expected Range for Feature $i$ is the count of values in column $i$ that are outside the specified min/max range, and Total Number of Rows is the total number of entries in the dataset.

*   **Cardinality Check:** For categorical features, this examines the number of unique values. Extremely low cardinality (e.g., only one unique value) or extremely high cardinality (e.g., unique values approaching the total number of rows) can indicate issues.

### Perform Data Quality Checks

1.  Ensure you have completed "Data Upload & Configuration" and applied your settings.
2.  Click the "Perform Data Quality Checks" button.
3.  The application will process the data according to your configurations and display the results.

### Interpreting the Results

The "Data Quality Assessment Results" table will show the status (PASS, WARN, FAIL) for each quality metric across your dataset and individual features, along with the calculated value and the thresholds used for comparison.

*   **Overall Data Quality Status:** This summarises the most severe issue found across all checks.

<aside class="positive">
Maya's data quality assessment reveals critical insights. The table clearly shows features flagged with 'WARN' or 'FAIL'. For example, `income` might have a 'FAIL' for missingness, indicating a significant portion of income data is absent. `credit_score` might show a 'FAIL' for type inconsistency due to non-numeric entries, which would break any numerical operations. `age` could have a 'WARN' for range violations, requiring a cleanup of outlier values.
</aside>

These findings directly inform Maya's next steps:
*   **Missing Data:** Maya must decide on an imputation strategy (e.g., mean, median, predictive imputation) or whether the column should be dropped.
*   **Type Inconsistencies:** Requires data cleaning to convert values to the correct type or remove corrupted entries.
*   **Range Violations:** Outliers need to be handled, either corrected, capped, or removed, to prevent skewed model learning.

By addressing these issues proactively, Maya ensures the credit approval model receives reliable data, preventing it from making decisions based on faulty inputs.

## 3. Bias Metric Computation: Ensuring Fairness in Data
Duration: 00:08:00

At FinTech Innovators Inc., ensuring fairness and avoiding discrimination in credit decisions is not just a regulatory requirement but a core ethical principle. Maya understands that biases present in the training data can be learned and amplified by models, leading to unfair outcomes for certain demographic groups. Before training the credit approval model, she must quantify any inherent biases within the new dataset. This assessment helps her identify if the raw data itself exhibits disparities in credit repayment outcomes across sensitive attributes like `marital_status` or `region`.

Since we are in a pre-training context (no model predictions yet), we will focus on measuring statistical parity and outcome disparities based on the *actual* target variable distributions across protected groups.

### Key Bias Metrics

*   **Demographic Parity Difference (DPD):** Measures the difference in the proportion of the favorable outcome (e.g., loan repaid) between an unprivileged group and a privileged group. A value close to 0 indicates demographic parity.
    $$ DPD = P(Y=1 | A_{\text{unprivileged}}) - P(Y=1 | A_{\text{privileged}}) $$
    where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.

*   **Disparate Impact Ratio (DIR):** Measures the ratio of the favorable outcome proportion for the unprivileged group to the privileged group. A value near 1 suggests no disparate impact. Values significantly below 1 (e.g., < 0.8) indicate the unprivileged group is less likely to receive the favorable outcome, while values significantly above 1 (e.g., > 1.25) indicate the unprivileged group is more likely.
    $$ DIR = \frac{{P(Y=1 | A_{\text{unprivileged}})}}{{P(Y=1 | A_{\text{privileged}})}} $$
    where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.

*   **Proxy True Positive Rate Gap (TPR Gap):** In a pre-training context, this can be interpreted as the difference in the *actual positive outcome rate* (prevalence of $Y=1$) between unprivileged and privileged groups. A value close to 0 indicates similar rates of positive outcomes for both groups in the raw data.
    $$ \text{{Proxy TPR Gap}} = P(Y=1 | A_{\text{unprivileged}}) - P(Y=1 | A_{\text{privileged}}) $$
    where $Y=1$ is the favorable outcome (e.g., loan repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.

*   **Proxy False Positive Rate Gap (FPR Gap):** Similarly, in a pre-training context, this can be interpreted as the difference in the *actual negative outcome rate* (prevalence of $Y=0$) between unprivileged and privileged groups. A value close to 0 indicates similar rates of negative outcomes for both groups in the raw data.
    $$ \text{{Proxy FPR Gap}} = P(Y=0 | A_{\text{unprivileged}}) - P(Y=0 | A_{\text{privileged}}) $$
    where $Y=0$ is the unfavorable outcome (e.g., loan not repaid) and $A$ denotes the sensitive attribute, with $A_{\text{unprivileged}}$ and $A_{\text{privileged}}$ representing the unprivileged and privileged groups, respectively.

### Compute Bias Metrics

1.  Ensure you have completed "Data Upload & Configuration" and applied your settings, including selecting sensitive attributes and defining protected groups.
2.  Click the "Compute Bias Metrics" button.
3.  The application will calculate the bias metrics for each sensitive attribute and display the results.

### Interpreting the Results

The "Bias Metrics Assessment Results" table will show the status (PASS, WARN, FAIL) for each bias metric across your selected sensitive attributes, along with the calculated value and the thresholds used for comparison.

*   **Overall Bias Status:** This summarises the most severe bias issue found.

<aside class="positive">
The bias metrics report provides Maya with quantitative evidence of fairness (or lack thereof) in the raw data. For instance, if the `region` attribute shows a Disparate Impact Ratio significantly below 1 (e.g., for the 'South' region), it indicates that customers from this region are less likely to have favorable credit outcomes ($Y=1$) in the dataset compared to the privileged 'North' region. This is a critical 'WARN' or 'FAIL' condition for FinTech Innovators Inc.
</aside>

Maya must now consider strategies to mitigate these biases *before* model training. This could involve:
*   **Data collection review:** Investigating if the data collection process itself introduced biases.
*   **Feature engineering:** Creating new features that might reduce reliance on sensitive attributes.
*   **Resampling techniques:** Over-sampling underrepresented groups or under-sampling overrepresented groups to balance the dataset.

These steps are crucial for FinTech Innovators Inc. to ensure equitable credit access and avoid legal or reputational risks.

## 4. Drift Detection: Monitoring Data Distribution Shifts
Duration: 00:07:00

FinTech Innovators Inc. operates in a dynamic financial market, where customer behaviors and economic conditions can change rapidly. Maya understands that a credit approval model trained on old data might become less accurate if the underlying data distribution shifts over time. This phenomenon, known as "data drift," can severely degrade model performance in production. To mitigate this risk, Maya needs to compare the new credit application dataset against a historical "baseline" dataset (the data the current production model was originally trained on).

### Population Stability Index (PSI)

The **Population Stability Index (PSI)** is a widely used metric to quantify data drift for numerical features. It measures how much a variable's distribution has changed from a baseline period to a current period.

**Population Stability Index (PSI):** For each feature, the PSI is calculated by:
1.  Dividing the range of the feature into several bins (e.g., 10 bins).
2.  Calculating the percentage of observations in each bin for both the baseline ($P_{{0i}}$) and the current ($P_i$) datasets.
3.  Summing the contributions from each bin:
    $$ \text{{PSI}} = \sum_{{i=1}}^{{N}} (P_i - P_{{0i}}) \times \ln\left(\frac{{P_i}}{{P_{{0i}}}}\right) $$
    where $N$ is the number of bins, $P_i$ is the percentage of observations in bin $i$ for the current dataset, and $P_{{0i}}$ is the percentage of observations in bin $i$ for the baseline dataset. A small constant (epsilon) is typically added to $P_i$ and $P_{{0i}}$ to avoid issues with zero values in the logarithm.

Common interpretations for PSI:
*   PSI < 0.1: No significant shift (PASS)
*   0.1 <= PSI < 0.25: Small shift (WARN)
*   PSI >= 0.25: Significant shift (FAIL)

### Detect Data Drift

1.  Ensure you have completed "Data Upload & Configuration" and applied your settings, and importantly, have uploaded a **Baseline Dataset**. If no baseline is provided, this step will be skipped.
2.  Click the "Detect Data Drift (PSI)" button.
3.  The application will calculate the PSI for numerical features and display the results.

### Interpreting the Results and Visualizations

The "Drift Detection Results (Population Stability Index - PSI)" table will show the PSI value and status (PASS, WARN, FAIL) for each numerical feature, along with the thresholds.

*   **Overall Drift Status:** This summarises the most severe drift issue found.

Below the table, "Visualizing Drift for Key Numerical Features" provides histograms comparing the distributions of selected features between your primary and baseline datasets. This visual comparison helps to intuitively understand the nature and extent of the detected drift.

<aside class="positive">
Maya examines the PSI results to detect significant shifts. For instance, if `income` or `credit_score` show a 'WARN' or 'FAIL' PSI value, it means the distribution of these critical features in the new dataset has changed notably from the baseline. The visualizations provide an intuitive understanding of these shifts. A shift in `income` distribution, for example, could indicate changing economic conditions or a different customer segment applying for credit.
</aside>

This insight is crucial for Maya:
*   **Model Re-training:** A significant drift (`FAIL`) in key features might necessitate immediate re-training of the credit approval model using the new data.
*   **Monitoring Strategy:** Even a 'WARN' suggests closer monitoring of the model's performance on these features in production.
*   **Business Context:** It prompts FinTech Innovators Inc. to investigate the business reasons behind the drift, informing both model development and business strategy.

By systematically identifying data drift, Maya helps FinTech Innovators Inc. maintain accurate and relevant models, preventing unexpected performance degradation in live environments.

## 5. Overall Readiness Decision and Comprehensive Reporting
Duration: 00:07:00

After conducting thorough data quality, bias, and drift assessments, Maya needs to synthesize all findings into a clear, deterministic decision regarding the dataset's readiness for model training. This final decision, along with a comprehensive report, is critical for communication with Model Validators and Risk/Compliance Partners at FinTech Innovators Inc. It provides a transparent, quantitative basis for proceeding or halting model development, ensuring accountability and adherence to governance standards.

The overall readiness decision follows a strict logic:
*   **DO NOT DEPLOY:** If any 'FAIL' condition is identified across data quality, bias, or drift metrics. This indicates severe issues that must be addressed before proceeding.
*   **PROCEED WITH MITIGATION:** If there are only 'WARN' conditions, but no 'FAIL's. This means the dataset has minor issues that can likely be mitigated (e.g., specific data cleaning, bias-aware modeling) while proceeding with development.
*   **PROCEED:** If all checks result in a 'PASS'. This indicates the dataset is robust and ready for model training without immediate, significant concerns.

### Generate Final Report and Export Artifacts

1.  Ensure you have completed all previous steps: "Data Upload & Configuration", "Data Quality", and "Bias Metrics". If you used a baseline, "Drift Detection" should also be complete.
2.  Click the "Generate Final Report and Export Artifacts" button.
3.  The application will consolidate all results, determine the overall readiness decision, and generate comprehensive reports.

### Review the Readiness Decision and Reports

*   **Overall Dataset Readiness Decision:** This prominent display will show the final verdict (DO NOT DEPLOY, PROCEED WITH MITIGATION, or PROCEED).
*   **Reports and Audit Artifacts:** The path to the generated reports folder and the bundled zip archive will be displayed.
*   **Download Audit Bundle (.zip):** A button will appear allowing you to download a zip file containing all reports and artifacts.
*   **Executive Summary:** A summary markdown report will be displayed directly in the application, providing a high-level overview of the findings.

<aside class="positive">
Maya has now generated a comprehensive suite of reports. The `overall_dataset_readiness_status` provides a definitive answer for her stakeholders. For FinTech Innovators Inc., a 'DO NOT DEPLOY' status (due to the simulated 'FAIL' conditions in our sample data) means Maya must halt the model development pipeline. She then shares the `session04_executive_summary.md` with the Model Validator and Risk/Compliance Partner. This markdown document provides a high-level overview, highlighting critical issues (e.g., "Feature `income`: Missingness: 25.00% (`FAIL`)", "Sensitive Attribute `region`: Disparate Impact Ratio: 0.6700 (`FAIL`)"), and the clear recommendation.
</aside>

The JSON reports (e.g., `data_quality_report.json`) provide the granular details necessary for deeper investigation by the technical team. This structured reporting ensures that FinTech Innovators Inc. maintains a clear audit trail for all data-related decisions and adheres to its internal governance framework.

### Exporting Audit Artifacts: Ensuring Traceability and Compliance

For FinTech Innovators Inc., regulatory compliance and internal auditability are paramount. Every decision related to model development, especially data quality, must be fully traceable and defensible. Maya's final, critical step is to bundle all generated reports, configurations, and an evidence manifest (containing SHA-256 hashes of all artifacts) into a secure, version-controlled zip archive. This ensures that the assessment results are immutable, tamper-evident, and readily available for future audits or reviews by risk and compliance teams.

**Evidence Manifest:** A record of all generated files, including their SHA-256 hash. The SHA-256 hash is a cryptographic checksum that uniquely identifies the content of a file. If even a single bit in the file changes, its SHA-256 hash will be drastically different.
$$ \text{{SHA-256 Hash}} = \text{{SHA256}}(\text{{file\_content}}) $$
where SHA-256 Hash is the cryptographic checksum, and file\_content is the entire binary content of the file.

This cryptographic integrity check confirms that the reports have not been altered since their generation.

<aside class="positive">
Maya has successfully created a zip archive containing all the assessment reports, the configuration snapshot, and a crucial `evidence_manifest.json` file. This manifest acts as a digital fingerprint for all generated documents. The verification step demonstrates how a Model Validator or Risk Partner at FinTech Innovators Inc. can confirm the integrity of any report by recomputing its hash and comparing it against the manifest. If any report were to be tampered with, the hash verification would fail, immediately flagging a potential issue.
</aside>

This robust export mechanism ensures that FinTech Innovators Inc. can confidently demonstrate compliance, provide auditable evidence of data quality and fairness assessments, and maintain strict governance over its ML models throughout their lifecycle. Maya's work is complete, providing FinTech Innovators Inc. with the objective evidence needed to make informed decisions about its credit approval model.
