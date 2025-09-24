# Credit Risk Capstone — Predicting Loan Default with CRISP-DM (R)

![R](https://img.shields.io/badge/R-4.2.x-276DC3?logo=r&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-CRISP--DM-4B5563)
![Focus](https://img.shields.io/badge/Focus-Credit%20Risk%20%7C%20Fair%20Lending-0EA5E9)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-black)

---

## Executive Summary

This project applies the **CRISP-DM** process to build a transparent, production-oriented **credit risk model** for predicting **loan default**. Using R (tidyverse + caret + pROC), I cleaned and prepared a 1,000-record applicant dataset, engineered features, and evaluated several algorithms. The final model is a **logistic regression** that balances accuracy and interpretability for business stakeholders.  

**Highlights**
- **Data**: 32 predictors after cleaning/encoding; moderate class imbalance.
- **Model**: Logistic Regression (binomial GLM), standardized numerics, one-hot encoded categoricals.
- **Test Metrics**: **ROC-AUC ≈ 0.817**, **Balanced Accuracy ≈ 68.9%**.  
  Confusion Matrix (example run): **TN=145, TP=37, FP=23, FN=35**.
- **Governance**: Sensitive fields excluded; fair-lending and monitoring considerations documented.
- **Delivery**: Executive slides + (optional) narrated video; reproducible R script and code revision.

---

## Quick Links

- 🎥 **Video Walkthrough (MP4)** → `[ADD VIDEO LINK HERE]`  
- 📊 **Slides, presentation ready (PPTX)** → `[ADD PPTX PATH HERE e.g., ./slides/Credit-Risk-Capstone.pptx]`  
- 🖥️ **Slides, with speaker notes (PDF)** → `./slides/690-Capstone-Final-Presentation.pdf`  
- 📝 **Full Report (PDF)** → [crisp-dm-capstone.pdf](./crisp-dm-capstone.pdf)  
- 🧪 **Code Revision (PDF)** → [code-revision.pdf](./code-revision.pdf)  




---

## Table of Contents
- [Project Context](#project-context)
- [Data Summary](#data-summary)
- [CRISP-DM Workflow](#crispdm-workflow)
  - [1) Business Understanding](#1-business-understanding)
  - [2) Data Understanding](#2-data-understanding)
  - [3) Data Preparation](#3-data-preparation)
  - [4) Modeling](#4-modeling)
  - [5) Evaluation](#5-evaluation)
  - [6) Deployment (Plan)](#6-deployment-plan)
- [Model Results](#model-results)
- [Ethics & Fair-Lending](#ethics--fair-lending)
- [Reproducibility (R)](#reproducibility-r)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Key Skills Demonstrated](#key-skills-demonstrated)
- [Limitations & Next Steps](#limitations--next-steps)
- [License](#license)
- [Tags](#tags)

---

## Project Context

**Goal:** Predict the probability that a credit applicant will **default** to improve underwriting decisions, reduce losses, and speed up reviews.  
**Why logistic regression?** It provides **probability outputs** and **interpretable coefficients**, supporting **model risk management** and **executive sign-off**.

---

## Data Summary

- **Scope:** ~1,000 historic applicants, 32 variables after selection/encoding.  
- **Types:** Numeric (e.g., AMOUNT, DURATION, AGE), categorical (e.g., SAV_ACCT, CHK_ACCT), binary flags.  
- **Notes:** Moderate class imbalance (defaults < non-defaults); no missingness in the working sample after cleaning.  
- **Not distributed:** Dataset is course-provided; paths are parameterized for local replication.

---

## CRISP-DM Workflow

### 1) Business Understanding
- Reduce default losses and manual review time by **scoring** incoming applications.
- Require **transparent**, **auditable**, and **bias-aware** modeling.

### 2) Data Understanding
- **EDA:** `head/str/summary`, descriptive stats with `psych::describe`, **corr heatmap** (`corrplot`).
- **Findings:** DURATION and AMOUNT moderately correlated; other predictors add complementary signal.

### 3) Data Preparation
- **Cleaning:** No duplicates; types verified; categoricals → factors.
- **Encoding:** One-hot for categoricals where needed.
- **Scaling:** Standardize numerics (e.g., `scale(AGE)`, `scale(AMOUNT)`).
- **Split:** **70/30 train/test** (seeded). *(Earlier planning noted 80/20; final code uses 70/30 for a larger test set.)*

### 4) Modeling
- **Tried/considered:** Naïve Bayes, Decision Trees (for feature intuition), Logistic Regression (primary).
- **Chosen:** **Logistic Regression (GLM binomial)** for interpretability + strong AUC.
- **Package stack:** `caret` (partition, evaluation), `pROC` (ROC/AUC).

### 5) Evaluation
- **Metrics:** ROC-AUC, Confusion Matrix, Precision/Recall, Balanced Accuracy.  
- **Representative results:** AUC ≈ **0.8168**; Balanced Accuracy ≈ **68.85%**.

### 6) Deployment (Plan)
- Export coefficients / scoring code; package as an API or batch scorer.
- **Monitoring:** drift, stability, threshold tuning, fairness checks, periodic retraining.

---

## Model Results

- **Confusion Matrix (test):** TN=145, TP=37, FP=23, FN=35  
- **AUC:** ~**0.817** (strong class separation)  
- **Balanced Accuracy:** ~**68.9%**  
- **Interpretation:** Savings/checking account indicators, scaled AMOUNT/AGE, and residency flags contribute materially; signs are directionally consistent with risk intuition.

---

## Ethics & Fair-Lending

- **Excluded** sensitive attributes (e.g., gender/race/marital status).  
- **Documented** variable rationale; retained only business-relevant predictors.  
- **Monitoring plan:** Periodic fairness diagnostics (e.g., outcome rates by non-protected operational segments), threshold reviews, and governance sign-offs.

---

## Reproducibility (R)

**Environment:** R 4.2.x, RStudio (Windows/macOS).  
**Key packages:** `readr`, `dplyr`, `ggplot2`, `corrplot`, `psych`, `caret`, `pROC`.

```r
# Install (if needed)
pkgs <- c("readr","dplyr","ggplot2","corrplot","psych","caret","pROC")
inst <- pkgs %in% rownames(installed.packages())
if (any(!inst)) install.packages(pkgs[!inst])

library(readr); library(dplyr); library(ggplot2)
library(corrplot); library(psych); library(caret); library(pROC)

# ---- Paths (edit as needed) ----
# data/ is not included in this repo (course dataset)
data_path <- "data/CreditRisk_Data.csv"

# Load
df <- read.csv(data_path)

# Minimal prep
df$SAV_ACCT <- as.factor(df$SAV_ACCT)
df$CHK_ACCT <- as.factor(df$CHK_ACCT)
df$FOREIGN  <- as.factor(df$FOREIGN)
df$AGE      <- scale(df$AGE)
df$AMOUNT   <- scale(df$AMOUNT)

set.seed(123)
idx <- createDataPartition(df$DEFAULT, p = 0.7, list = FALSE)
train <- df[idx, ]; test <- df[-idx, ]

# Logistic regression
m <- glm(DEFAULT ~ SAV_ACCT + CHK_ACCT + AGE + AMOUNT + FOREIGN,
         data = train, family = "binomial")

# Predictions & evaluation
p  <- predict(m, newdata = test, type = "response")
y  <- ifelse(p > 0.5, 1, 0)
cm <- table(Predicted = y, Actual = test$DEFAULT); print(cm)

roc_obj <- roc(test$DEFAULT, p)
plot(roc_obj, main = "ROC Curve (Logit)")
auc(roc_obj)

```
---

> The full code-revision script includes richer comments, caret confusion matrix, and notes on future threshold tuning and SMOTE.

---

## Repository Structure  

credit-risk-capstone/
├─ README.md
├─ slides/
│ ├─ 690-Capstone-Final-Presentation.pdf # uploaded
│ └─ Credit-Risk-Capstone.pptx # ADD your PPTX
├─ video/
│ └─ credit-risk-capstone-demo.mp4 # ADD your screen-record video
├─ docs/
│ └─ Credit-Risk-Capstone-Report.pdf # ADD your full write-up PDF
├─ code/
│ ├─ credit-risk-model.R # (optional) clean main script
│ └─ code-revision.R # ADD your 8-1 Programming Revision
└─ LICENSE

---


> Update the **Quick Links** section at the top with the exact paths you use here.  

---

## Technologies  

- **R / RStudio**  
- **Methods:** Logistic Regression (GLM), Decision Trees (for exploration), Naïve Bayes (considered)  
- **Packages:** `caret`, `pROC`, `dplyr`, `ggplot2`, `corrplot`, `psych`  
- **Practices:** standardization, encoding, holdout validation, ROC/AUC, governance  

---

## Key Skills Demonstrated  

- **End-to-end CRISP-DM**: from business framing → deployment plan  
- **Risk modeling**: probability of default with interpretable coefficients  
- **Data preparation**: cleaning, encoding, scaling, leakage avoidance  
- **Model evaluation**: ROC-AUC, confusion matrix, class-imbalance awareness  
- **Governance**: ethical feature selection, fairness monitoring, documentation  
- **Communication**: executive-ready slides, annotated code, reproducibility  

---

## Limitations & Next Steps  

- **Imbalance**: experiment with SMOTE, class weights, and threshold tuning  
- **Modeling**: benchmark against regularized GLM, Gradient Boosting, Random Forest  
- **Validation**: add cross-validation, calibration curves, PSI/KS in monitoring  
- **Deployment**: package scoring function/API; integrate with CRM; build dashboards for drift & fairness  

---

## License  

Released under the **MIT License**. See [LICENSE](./LICENSE).  

---

## Tags  

`credit-risk` `crisp-dm` `logistic-regression` `roc-auc` `caret`  
`fair-lending` `model-governance` `class-imbalance` `r` `data-science` `portfolio`  


