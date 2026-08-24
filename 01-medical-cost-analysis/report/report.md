---
title: "Medical Cost Prediction Using Linear Regression: PDAN8411 POE Part 1"
author: "Liam Johnie | ST10536072 | 23 April 2026"
---

# Summary Overview

This report presents a proof-of-concept linear regression model for predicting annual medical charges, demonstrating the modelling methodology for a South African medical aid scheme. The analysis uses a publicly available US health insurance dataset of 1,338 rows (Choi, 2018), reduced to 1,337 after removing one duplicate. Exploratory data analysis identified smoker status as the dominant cost driver: median charges for smokers ($34,456) are 4.7 times higher than for non-smokers ($7,346), with a Pearson correlation of 0.79 with annual charges. Backward elimination on a statsmodels OLS model retained four predictors - age, bmi, children, and smoker_yes - achieving an OLS R-squared of 0.750. Residual diagnostics on the raw-charges scikit-learn model revealed violations of the homoscedasticity and normality assumptions (target skewness 1.5154). A log transformation reduced skewness to -0.0898 and produced better-behaved residuals. The log model was selected as the final model, achieving a USD-space test R-squared of 0.7192 and a mean absolute error of $3,871.05.

# 1. Introduction

Healthcare cost prediction is a core task for medical aid schemes seeking to set premiums accurately and manage reserves. Linear regression offers a transparent, interpretable starting point: its coefficients can be explained directly to actuaries without specialist machine learning knowledge (Malehi, Pourmotahari and Angali, 2015). This report implements the full pipeline - data cleaning, exploratory analysis, OLS feature selection, scikit-learn model fitting, and residual diagnostics - using the Kaggle insurance dataset specified in the PDAN8411 POE brief (Choi, 2018).

A key limitation applies throughout. This dataset represents US private insurance pricing in United States Dollars. The intended application is a South African medical aid scheme, which operates in South African Rand within a different regulatory framework and healthcare system (Council for Medical Schemes, 2024). Population health profiles and cost structures differ between the two countries, so this analysis is a methodological proof-of-concept; retraining on South African claims data is required before production deployment.

# 2. Dataset Overview

The dataset contains 1,338 rows and 7 columns. The raw dataset contains 1,338 rows (Choi, 2018). After removing one duplicate row during EDA, 1,337 rows remain for analysis. No missing values were found in any column.

| Column   | Type        | Description                                |
|:---------|:------------|:-------------------------------------------|
| age      | Integer     | Age of primary beneficiary (years)         |
| sex      | Categorical | Sex of policyholder (male / female)        |
| bmi      | Float       | Body mass index (kg/m^2)                   |
| children | Integer     | Number of dependants covered               |
| smoker   | Categorical | Smoker status (yes / no)                   |
| region   | Categorical | US geographic region (four categories)     |
| charges  | Float       | Annual medical charges (USD)               |

*Table 1: Dataset variables and types (Choi, 2018).*

The three categorical variables were one-hot encoded with `drop_first=True`, producing five binary dummy variables. Data loading and manipulation used pandas (pandas development team, 2024). The target charges has a mean of $13,279 and a standard deviation of $12,110.

# 3. Exploratory Data Analysis

The EDA examined missing values, duplicates, univariate distributions, outliers, and correlations (GeeksforGeeks, 2024). Visualisations were produced using Matplotlib (The Matplotlib development team, 2024) and seaborn (Waskom, 2021).

**Data quality.** No missing values were found. One duplicate was removed, leaving 1,337 rows. BMI has nine values above its upper inter-quartile fence (47.32, maximum 53.13); these were retained as plausible extreme values. Charges has 139 values above $34,525.

**Distribution of charges.** Figure 1 shows that charges are right-skewed (skewness 1.5154); the mean ($13,279) substantially exceeds the median ($9,386). This skewness motivates the log transformation applied in Section 5.

**Smoker effect.** Figure 2 shows that median charges for smokers ($34,456) are 4.7 times higher than for non-smokers ($7,346), a gap of $27,110. This is consistent with the well-documented relationship between tobacco use and elevated healthcare costs (WHO, 2025), and is confirmed by a Pearson correlation of 0.79 between smoker_yes and charges.

**BMI-smoker interaction.** Figure 3 reveals that non-smokers cluster at low charges regardless of BMI, while smokers above BMI 30 form a distinct high-charges group. This non-additive pattern limits a standard additive linear model.

**Correlations.** Figure 4 shows the Pearson heatmap. smoker_yes dominates (0.79), followed by age (0.30) and bmi (0.20). The highest inter-predictor correlation among the final four features is 0.11 (age and bmi), indicating multicollinearity is not a concern.

\begin{figure}[h!]
\centering
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_charges_histogram.png}
  \\\small\textit{Figure 1: Distribution of charges (skewness 1.5154). The right tail is driven by high-cost smoker cases.}
\end{minipage}
\hfill
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_smoker_boxplot.png}
  \\\small\textit{Figure 2: Median charges for smokers (\$34,456) are 4.7 times higher than for non-smokers (\$7,346).}
\end{minipage}
\end{figure}

\begin{figure}[h!]
\centering
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_bmi_smoker_scatter.png}
  \\\small\textit{Figure 3: BMI vs charges by smoker status. Smokers above BMI 30 form a distinct high-charges group.}
\end{minipage}
\hfill
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_correlation_heatmap.png}
  \\\small\textit{Figure 4: Pearson correlation heatmap. smoker\_yes leads (0.79); maximum inter-predictor correlation is 0.11.}
\end{minipage}
\end{figure}

# 4. Data Splitting

The 1,337-row dataset was split 80/20 into training (1,069 rows) and test (268 rows) using scikit-learn's `train_test_split` with `random_state=42` (scikit-learn developers, 2024b). An 80/20 split was chosen because the dataset is small; maximising training data reduces estimation variance without leaving an unreasonably small test set. The same `random_state=42` was applied for both the raw-charges and log-charges models so that the same 268 rows form the test set in both evaluations, making metric comparisons valid.

# 5. Model Creation and Fitting

**Feature selection.** After one-hot encoding, eight candidate predictors were available. A statsmodels OLS model was fitted on all eight to obtain p-values (statsmodels developers, 2024a). Backward elimination (threshold p > 0.05) removed four in sequence: sex_male (p = 0.698), region_northwest (p = 0.465), region_southwest (p = 0.058), and region_southeast (p = 0.136). The four retained predictors are age, bmi, children, and smoker_yes. The final OLS model achieves an R-squared of 0.750 and an adjusted R-squared of 0.749. Adjusted R-squared is reported here because comparing nested models during feature selection is precisely the use case for which it was designed (Malehi et al., 2015). VIF confirmed no multicollinearity: all VIF values are approximately 1.0, with a maximum of 1.0145 for age (statsmodels developers, 2024b).

**Raw-charges model.** A scikit-learn `LinearRegression` with default settings was fitted on the 1,069 training rows (scikit-learn developers, 2024a). Training R-squared was 0.7292. Residual diagnostics on the test set revealed heteroscedasticity and non-normal residuals, both expected given the bimodal cost structure created by smoker status.

**Log-charges model.** A natural log transformation was applied to charges, reducing skewness from 1.5154 to -0.0898. Log-transforming a right-skewed outcome before fitting a linear model is a standard technique for improving assumption compliance (Malehi et al., 2015). The model was retrained on the same training features and identical train-test split.

# 6. Model Evaluation

Metrics were computed on the 268-row test set for both models (scikit-learn developers, 2024c). Adjusted R-squared is not reported for the scikit-learn LinearRegression models; it was reported for the OLS feature-selection stage (adjusted R-squared 0.749) where comparing nested models is the primary goal.

| Metric                 | Raw-Charges Model | Log-Charges Model |
|:-----------------------|------------------:|------------------:|
| R-squared (USD, test)  |            0.8046 |            0.7192 |
| RMSE (USD, test)       |         \$5,992.88 |         \$7,183.70 |
| MAE (USD, test)        |         \$4,198.59 |         \$3,871.05 |
| Homoscedasticity       |          Violated |          Improved |
| Normality of residuals |          Violated |          Improved |

*Table 2: Comparison of raw-charges and log-charges LinearRegression models on the 268-row test set.*

**Raw-charges model.** The raw model achieves a test R-squared of 0.8046 and RMSE of $5,992.88. Figure 5 shows two distinct residual clusters corresponding to non-smoker and smoker subpopulations; residuals in the upper cluster are predominantly positive, indicating systematic underestimation of high-cost smoker cases. The QQ plot (Figure 6), produced using scipy.stats.probplot (SciPy developers, 2024), confirms heavy tails (residual skewness 1.25), violating the normality assumption. Both violations are driven by the bimodal cost structure.

**Log-charges model.** Figure 7 shows the log-model residuals: the two-cluster pattern is reduced and spread is more uniform. The QQ plot (Figure 8) places points closer to the reference line. The log model's test MAE of $3,871.05 is $327.54 lower than the raw model's, indicating smaller typical prediction errors. However, the log model RMSE of $7,183.70 exceeds the raw model's $5,992.88 because the exp() back-transformation amplifies large individual errors.

**Final model selection.** The log-charges model is selected as the final model. It produces better-behaved residuals, improving compliance with the statistical assumptions required for valid linear regression inference (GeeksforGeeks, 2024). Its lower MAE ($3,871.05 vs $4,198.59) corresponds to smaller typical prediction errors - the more policy-relevant metric for pricing the majority of scheme members. The higher RMSE is a known trade-off of the exp() back-transformation and should be raised with the client.

\begin{figure}[h!]
\centering
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_raw_residuals.png}
  \\\small\textit{Figure 5: Residuals vs fitted, raw model. Two clusters and upward bias in the high-charges group indicate heteroscedasticity.}
\end{minipage}
\hfill
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_raw_qq.png}
  \\\small\textit{Figure 6: QQ plot, raw model. Heavy upper tail confirms non-normality (residual skewness 1.25).}
\end{minipage}
\end{figure}

\begin{figure}[h!]
\centering
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_log_residuals.png}
  \\\small\textit{Figure 7: Residuals vs fitted, log model. More uniform spread indicates improved homoscedasticity relative to Figure 5.}
\end{minipage}
\hfill
\begin{minipage}[b]{0.47\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/fig_log_qq.png}
  \\\small\textit{Figure 8: QQ plot, log model. Points closer to the diagonal indicate improved but not fully satisfied normality.}
\end{minipage}
\end{figure}

# 7. Conclusion

This study demonstrates a complete linear regression workflow for medical cost prediction. Smoker status is by far the dominant predictor, with a Pearson correlation of 0.79 and a median charge gap of $27,110. Backward elimination selected four statistically significant features (age, bmi, children, smoker_yes). The log-charges model was selected as the final model: USD-space test R-squared 0.7192, test MAE $3,871.05, with substantially improved residual diagnostics relative to the raw-charges baseline.

For application to a South African medical aid scheme, three limitations must be addressed before deployment. First, the dataset is US-based; South African medical aid costs and member demographics differ substantially (Council for Medical Schemes, 2024), and retraining on local claims data is necessary. Second, the BMI-smoker interaction identified in EDA is not captured by the additive model; an interaction term or a non-linear method may improve accuracy. Third, evaluation relies on a single 80/20 split; k-fold cross-validation would provide a more robust out-of-sample performance estimate.

# References

Choi, M., 2018. *Medical Cost Personal Datasets*. [online] Available at: <https://www.kaggle.com/datasets/mirichoi0218/insurance> [Accessed 23 April 2026].

Council for Medical Schemes, 2024. *CMS Annual Report 2023/24*. [online] Available at: <https://www.medicalschemes.co.za/cms-annual-report-2023-24/> [Accessed 23 April 2026].

GeeksforGeeks, 2024. *Assumptions of Linear Regression*. [online] Available at: <https://www.geeksforgeeks.org/assumptions-of-linear-regression/> [Accessed 23 April 2026].

Malehi, A.S., Pourmotahari, F. and Angali, K.A., 2015. Statistical models for the analysis of skewed healthcare cost data: a simulation study. *Health Economics Review*, 5, article 11. Available at: <https://link.springer.com/article/10.1186/s13561-015-0045-7> [Accessed 23 April 2026].

The Matplotlib development team, 2024. *matplotlib.pyplot*. [online] Available at: <https://matplotlib.org/stable/api/pyplot_summary.html> [Accessed 23 April 2026].

pandas development team, 2024. *pandas.DataFrame*. [online] Available at: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html> [Accessed 23 April 2026].

scikit-learn developers, 2024a. *sklearn.linear_model.LinearRegression*. [online] Available at: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html> [Accessed 23 April 2026].

scikit-learn developers, 2024b. *sklearn.model_selection.train_test_split*. [online] Available at: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html> [Accessed 23 April 2026].

scikit-learn developers, 2024c. *Metrics and scoring: quantifying the quality of predictions*. [online] Available at: <https://scikit-learn.org/stable/modules/model_evaluation.html> [Accessed 23 April 2026].

SciPy developers, 2024. *scipy.stats.probplot*. [online] Available at: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html> [Accessed 23 April 2026].

statsmodels developers, 2024a. *statsmodels.regression.linear_model.OLS*. [online] Available at: <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html> [Accessed 23 April 2026].

statsmodels developers, 2024b. *statsmodels.stats.outliers_influence.variance_inflation_factor*. [online] Available at: <https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html> [Accessed 23 April 2026].

Waskom, M., 2021. *seaborn: statistical data visualization*. [online] Available at: <https://seaborn.pydata.org/> [Accessed 23 April 2026].

WHO, 2025. *Tobacco*. [online] Available at: <https://www.who.int/news-room/fact-sheets/detail/tobacco> [Accessed 23 April 2026].
