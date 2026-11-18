# Model Card: Claims Fraud Triage (XGBoost)

## Intended use
Rank auto insurance claims by fraud risk so investigators can prioritise referrals. **Decision support only**: a high score triggers human review, never an automatic denial.

## Training data
1,000 public auto claims (2015 incidents) with 24.7% labelled as fraud. Identifiers (policy number, ZIP, street address) are excluded. The dataset is small and partly synthetic.

## Evaluation
Stratified 75/25 split. Model selection and threshold tuning were done on the training set with 5-fold CV; the test set was used once.

| Test metric | Value |
|---|---|
| ROC-AUC | 0.84 |
| PR-AUC | 0.56 |
| Recall @ threshold 0.13 | 0.89 |
| Precision @ threshold 0.13 | 0.47 |

## Known risks and limitations
- **Suspected label leakage:** `insured_hobbies` (chess, cross-fit) is among the top drivers with no causal rationale. Ablation: removing it drops CV ROC-AUC from 0.85 to 0.76 (PR-AUC 0.59 to 0.49). Treat 0.76 as the realistic estimate, and remove the feature before any real use.
- **Fairness:** `insured_sex`, `age`, `insured_education_level` and `insured_occupation` are model inputs. In a regulated setting these must be reviewed for proxy discrimination (e.g. comparing alert rates across groups) and removed where they aren't justified.
- **Small sample:** 62 frauds in the test set, so the metrics have wide confidence intervals.
- **Cost assumptions** behind the threshold are illustrative and must be replaced with real leakage and investigation costs.

## Monitoring (recommended)
Track drift in claim amounts and severity mix, alert volume per week, and SIU-confirmed precision. Retrain when precision drops below an agreed floor.
