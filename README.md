# Liam Johnie | Data Analytics Portfolio

This portfolio contains selected data analytics and machine learning projects completed during my Postgraduate Diploma in Data Analytics.

The projects demonstrate practical work across exploratory data analysis, statistical modelling, machine learning, natural language processing, model evaluation and data-driven business recommendations.

## Projects

### 1. Medical Cost Prediction Using Linear Regression

[View project](./01-medical-cost-analysis)

A proof-of-concept regression analysis investigating factors associated with annual medical insurance charges.

Key work completed:

* Cleaned and explored a public health insurance dataset.
* Performed exploratory data analysis covering distributions, correlations, outliers and data quality.
* Encoded categorical variables for modelling.
* Used statsmodels OLS and backward elimination for feature selection.
* Tested multicollinearity using variance inflation factors.
* Built and compared raw-target and log-transformed Linear Regression models.
* Evaluated model performance using R-squared, RMSE and MAE.
* Assessed linear regression assumptions using residual and Q-Q diagnostics.
* Identified smoking status as the strongest cost driver in the dataset.

Final model:

The log-transformed model was selected because it produced improved residual behaviour and a lower typical prediction error, with a test MAE of USD 3,871.05 and test R-squared of 0.7192.

Skills demonstrated:

`Python` `Jupyter Notebook` `pandas` `scikit-learn` `statsmodels` `Matplotlib` `seaborn` `SciPy` `Linear Regression` `EDA` `Feature Selection` `Regression Diagnostics`

---

### 2. Cervical Cancer Risk Classification

[View project](./02-cervical-cancer-classification)

A proof-of-concept binary classification model designed to identify higher-risk cervical cancer cases for prioritised clinical review.

Key work completed:

* Cleaned a clinical dataset containing missing values, duplicate observations and highly sparse variables.
* Prevented target leakage by separating diagnostic outcome variables from upstream risk factors.
* Addressed severe class imbalance.
* Used stratified train-test splitting and training-only feature scaling.
* Applied backward elimination using statsmodels Logistic Regression.
* Built and compared baseline Logistic Regression, class-balanced Logistic Regression and Random Forest models.
* Tuned regularisation using GridSearchCV and stratified cross-validation.
* Evaluated models using recall, precision, F1, ROC AUC and confusion matrices.
* Prioritised positive-class recall because false negatives were the most important error for the proposed triage use case.

Final model:

A class-balanced Logistic Regression model with `C = 0.1` correctly identified 7 of 11 positive test cases, achieving positive-class recall of 0.6364.

Skills demonstrated:

`Python` `Jupyter Notebook` `pandas` `NumPy` `scikit-learn` `statsmodels` `Logistic Regression` `Random Forest` `GridSearchCV` `Cross-Validation` `Class Imbalance` `Classification Metrics`

---

### 3. Medical Aid Customer Review Analysis

[View project](./03-medical-aid-customer-review-analysis)

A natural language processing project combining topic modelling and sentiment classification to identify the main drivers of customer dissatisfaction in insurance reviews.

Key work completed:

* Cleaned and prepared a large text dataset for natural language processing.
* Removed duplicate reviews and personal-name information.
* Normalised and tokenised text while preserving negation terms important for sentiment.
* Used VADER to derive initial sentiment labels and manually reviewed label quality.
* Used CountVectorizer with Latent Dirichlet Allocation to discover customer-review topics.
* Compared topic solutions and selected an eight-topic model using coherence and interpretability.
* Used TF-IDF features with unigrams and bigrams for supervised sentiment modelling.
* Compared Multinomial Naive Bayes, Logistic Regression and Linear SVC classifiers.
* Tuned the selected Logistic Regression pipeline using GridSearchCV.
* Performed confusion-matrix, coefficient and error analysis.
* Combined topic modelling and sentiment analysis to identify which customer concerns produced the highest levels of dissatisfaction.

Final model:

The tuned TF-IDF and Logistic Regression pipeline achieved a held-out macro-F1 of 0.798 and negative-class recall of approximately 83 percent.

Business finding:

Policy, contact and communication problems were identified as the highest-priority concern, with a 23.9 percent negative sentiment rate and the largest estimated volume of negative reviews.

Skills demonstrated:

`Python` `Jupyter Notebook` `pandas` `NumPy` `scikit-learn` `NLP` `TF-IDF` `LDA Topic Modelling` `VADER Sentiment Analysis` `Logistic Regression` `GridSearchCV` `Text Classification` `Matplotlib` `seaborn`

---

## Technical Skills Demonstrated

Across these projects I have applied:

* Python for data analysis and modelling
* Jupyter Notebook for reproducible analytical workflows
* pandas and NumPy for data preparation and transformation
* scikit-learn for machine learning pipelines and model evaluation
* statsmodels for statistical modelling and feature selection
* Matplotlib and seaborn for data visualisation
* Regression and classification modelling
* Exploratory data analysis
* Feature engineering and feature selection
* Cross-validation and hyperparameter tuning
* Handling of imbalanced datasets
* Natural language processing
* TF-IDF text representation
* Topic modelling
* Sentiment classification
* Model diagnostics and error analysis
* Translation of analytical findings into business recommendations

## Analytical Approach

My projects focus on more than producing a model score. I aim to understand the data, choose evaluation measures appropriate to the problem, test model limitations and interpret results in the context in which they would be used.

The projects in this portfolio are academic proof-of-concept analyses. Where public proxy datasets were used, the reports explicitly identify the limitations of applying those results to a different population or operational environment.

## About Me

I hold a Bachelor of Commerce in Supply Chain Management and am currently completing a Postgraduate Diploma in Data Analytics.

My interests include data analytics, statistical modelling, machine learning, business intelligence and the practical application of data to business decision-making.
