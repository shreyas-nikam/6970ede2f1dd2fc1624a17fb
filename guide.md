# Lab Guide: Data Quality, Bias & Drift Assessment Dashboard

## Case Context

FinTech Innovators Inc. is a rapidly growing financial services company facing increasing regulatory scrutiny around algorithmic fairness in credit decisions. Last quarter, a competitor received a $5M fine for discriminatory lending practices traced back to biased training data. Your leadership team has mandated a "data-first" governance framework: no ML model enters production without documented data quality and fairness assessments.

You've been assigned to evaluate a new credit application dataset before the team invests $200K in model development and infrastructure. Historical data shows that 30% of model failures stem from preventable data quality issues discovered post-deployment. Your assessment will determine whether to proceed, pause for data remediation, or halt the project entirely.

## Your Role

You are Maya, a Senior ML Engineer at FinTech Innovators Inc. Today, you're conducting a pre-training data risk assessment—a critical gate in your organization's ML development lifecycle. Your stakeholders include Model Validators who need quantitative evidence, Risk/Compliance Partners who require audit trails, and executives who need clear go/no-go decisions. Your analysis will directly impact business strategy, model investment decisions, and regulatory compliance posture.

## What You Will Do

- Load and configure datasets with business-relevant quality and fairness thresholds
- Execute systematic data quality checks to identify missing values, duplicates, type errors, and anomalies
- Compute bias metrics across protected attributes to quantify fairness gaps before model training
- Detect distribution drift between current and historical data to assess model relevance
- Generate an executive summary with a deterministic readiness decision and exportable audit artifacts

## Step-by-Step Instructions

1. **Launch the Application**
   - Start the Streamlit dashboard
   - Review the sidebar navigation and current assessment status indicators

2. **Data Upload & Configuration** (Page 1)
   1. Select your use case scenario (Credit, Healthcare, or Fraud)
   2. Click "Load Sample Dataset" to populate the dashboard
   3. Review the primary and baseline dataset previews
   4. Configure assessment parameters:
      - Verify the target column (e.g., `loan_repaid`)
      - Confirm sensitive attributes (e.g., `marital_status`, `region`)
   5. Expand "Define Protected Groups" and review privileged/unprivileged group definitions
   6. (Optional) Expand "Override Default Thresholds" to adjust warning/failure thresholds
   7. Click "Apply Configuration" and confirm success

3. **Data Quality Assessment** (Page 2)
   1. Click "Perform Data Quality Checks"
   2. Examine the results table for each feature's quality metrics
   3. Identify any WARN or FAIL statuses for missingness, type inconsistencies, range violations, or cardinality issues
   4. Note specific features requiring remediation

4. **Bias Metrics Analysis** (Page 3)
   1. Click "Compute Bias Metrics"
   2. Review the overall bias status and sensitive attribute breakdown
   3. Analyze Demographic Parity Difference, Disparate Impact Ratio, and TPR/FPR Gaps
   4. Identify which protected groups show unfair treatment patterns

5. **Drift Detection** (Page 4)
   1. Click "Detect Data Drift (PSI)"
   2. Examine PSI values for numerical features
   3. Review distribution visualizations comparing primary vs. baseline datasets
   4. Determine if significant population shifts require model retraining

6. **Generate Final Report** (Page 5)
   1. Click "Generate Final Report and Export Artifacts"
   2. Review the overall readiness decision: PROCEED, PROCEED WITH MITIGATION, or DO NOT DEPLOY
   3. Read the executive summary
   4. Download the audit bundle (.zip) containing all reports and evidence

## What This Lab Teaches

- **Pre-deployment due diligence**: Assessing data *before* costly model training prevents wasted resources and regulatory risk
- **Quantitative fairness evaluation**: Moving beyond intuition to measure bias systematically using established metrics (DPD, DIR, TPR/FPR gaps)
- **Data drift monitoring**: Understanding that static models degrade when underlying data distributions shift over time
- **Governance-ready documentation**: Creating audit trails that satisfy model validators, compliance teams, and regulators
- **Risk-based decision frameworks**: Translating technical metrics into clear business decisions with deterministic thresholds

## Discussion Questions

1. If your assessment shows "PROCEED WITH MITIGATION" due to warnings on the `income` feature (15% missingness) and `region` attribute (DIR = 0.82), what specific remediation steps would you propose before model training? What are the trade-offs?

2. Your baseline dataset is 18 months old, and drift analysis shows significant PSI values (0.28) for `credit_score`. How would you defend a recommendation to delay model deployment for data re-collection versus proceeding with retraining on current data?

3. A business stakeholder argues that removing the `marital_status` feature entirely will solve bias concerns. Using concepts from this lab, explain why this approach may be insufficient and what additional steps are necessary.

## Takeaway

Data quality and fairness are not post-hoc validations—they are foundational prerequisites for responsible AI that protect both business value and societal trust.
