"""
generate_figures.py
Produces 8 PNG figures for the PDAN8411 POE Part 1 report.
Run from F:\\Projects\\my-project\\report\\
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
DPI = 200
FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')

# ── Load and preprocess (matches notebook exactly) ────────────────────────────
df = pd.read_csv('insurance.csv')
print(f'Raw rows: {len(df)}')

df = df.drop_duplicates().reset_index(drop=True)
print(f'After deduplication: {len(df)} rows')

df_enc = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
for col in df_enc.select_dtypes(include='bool').columns:
    df_enc[col] = df_enc[col].astype(int)

features = ['age', 'bmi', 'children', 'smoker_yes']
X_selected = df_enc[features]
y = df_enc['charges']
y_log = np.log(y)

# 80/20 split with same random_state for both models
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=RANDOM_STATE)
_, _, y_train_log, y_test_log = train_test_split(
    X_selected, y_log, test_size=0.2, random_state=RANDOM_STATE)

print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

# Fit raw model
raw_model = LinearRegression()
raw_model.fit(X_train, y_train)
y_test_pred_raw = raw_model.predict(X_test)
residuals_raw = y_test.values - y_test_pred_raw

# Fit log model
log_model = LinearRegression()
log_model.fit(X_train, y_train_log)
y_test_pred_log = log_model.predict(X_test)
residuals_log = y_test_log.values - y_test_pred_log

# ── Print summary stats for verification ─────────────────────────────────────
print(f'\n--- Summary stats ---')
print(f'charges skewness (raw): {df["charges"].skew():.4f}')
print(f'charges skewness (log): {y_log.skew():.4f}')
print(f'smoker median charges:  ${df[df["smoker"]=="yes"]["charges"].median():.0f}')
print(f'non-smoker median:      ${df[df["smoker"]=="no"]["charges"].median():.0f}')
print(f'raw model train R2:     {r2_score(y_train, raw_model.predict(X_train)):.4f}')
print(f'raw model test  R2:     {r2_score(y_test, y_test_pred_raw):.4f}')
print(f'raw model test  RMSE:   {np.sqrt(mean_squared_error(y_test, y_test_pred_raw)):.2f}')
print(f'raw model test  MAE:    {mean_absolute_error(y_test, y_test_pred_raw):.2f}')
y_test_pred_usd = np.exp(log_model.predict(X_test))
y_test_act_usd  = np.exp(y_test_log)
print(f'log model test  RMSE (USD): {np.sqrt(mean_squared_error(y_test_act_usd, y_test_pred_usd)):.2f}')
print(f'log model test  MAE  (USD): {mean_absolute_error(y_test_act_usd, y_test_pred_usd):.2f}')
print(f'raw residual skewness: {pd.Series(residuals_raw).skew():.4f}')
print(f'log residual skewness: {pd.Series(residuals_log).skew():.4f}')

# ── Figure 1: Charges histogram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['charges'], kde=True, ax=ax, color='steelblue')
ax.set_title('Distribution of Annual Medical Charges')
ax.set_xlabel('Annual Charges (USD)')
ax.set_ylabel('Count')
ax.annotate(f'Skewness = {df["charges"].skew():.4f}', xy=(0.67, 0.88),
            xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_charges_histogram.png'), dpi=DPI)
plt.close()
print('Saved fig_charges_histogram.png')

# ── Figure 2: Correlation heatmap ─────────────────────────────────────────────
heatmap_cols = ['age', 'bmi', 'children', 'smoker_yes', 'charges']
corr_matrix = df_enc[heatmap_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, square=True)
ax.set_title('Pearson Correlation Heatmap (Key Features)')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_correlation_heatmap.png'), dpi=DPI)
plt.close()
print('Saved fig_correlation_heatmap.png')

# ── Figure 3: Smoker boxplot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df, x='smoker', y='charges', ax=ax,
            order=['no', 'yes'], palette=['steelblue', 'coral'])
ax.set_title('Annual Charges by Smoker Status')
ax.set_xlabel('Smoker Status')
ax.set_ylabel('Annual Charges (USD)')
ax.set_xticklabels(['Non-smoker', 'Smoker'])

medians = df.groupby('smoker')['charges'].median()
for i, (label, order) in enumerate([('no', 0), ('yes', 1)]):
    ax.text(order, medians[label] + 400, f'Median: ${medians[label]:,.0f}',
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_smoker_boxplot.png'), dpi=DPI)
plt.close()
print('Saved fig_smoker_boxplot.png')

# ── Figure 4: BMI-smoker scatter ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for label, color in [('no', 'steelblue'), ('yes', 'coral')]:
    subset = df[df['smoker'] == label]
    ax.scatter(subset['bmi'], subset['charges'], alpha=0.4,
               color=color, label='Non-smoker' if label == 'no' else 'Smoker', s=20)
ax.set_title('BMI vs Annual Charges by Smoker Status')
ax.set_xlabel('BMI')
ax.set_ylabel('Annual Charges (USD)')
ax.legend(title='Smoker Status')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_bmi_smoker_scatter.png'), dpi=DPI)
plt.close()
print('Saved fig_bmi_smoker_scatter.png')

# ── Figure 5: Raw model residuals ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test_pred_raw, residuals_raw, alpha=0.5, s=20, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label='Zero line')
ax.set_title('Residuals vs Fitted Values - Raw Model (Test Set)')
ax.set_xlabel('Fitted Values (Predicted Charges, USD)')
ax.set_ylabel('Residuals (USD)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_raw_residuals.png'), dpi=DPI)
plt.close()
print('Saved fig_raw_residuals.png')

# ── Figure 6: Raw model QQ ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
stats.probplot(residuals_raw, dist='norm', plot=ax)
ax.set_title('QQ Plot of Residuals - Raw Model (Test Set)')
ax.get_lines()[0].set(markersize=4, alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_raw_qq.png'), dpi=DPI)
plt.close()
print('Saved fig_raw_qq.png')

# ── Figure 7: Log model residuals ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test_pred_log, residuals_log, alpha=0.5, s=20, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label='Zero line')
ax.set_title('Residuals vs Fitted Values - Log Model (Test Set)')
ax.set_xlabel('Fitted Values (Predicted Log Charges)')
ax.set_ylabel('Residuals (Log Space)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_log_residuals.png'), dpi=DPI)
plt.close()
print('Saved fig_log_residuals.png')

# ── Figure 8: Log model QQ ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
stats.probplot(residuals_log, dist='norm', plot=ax)
ax.set_title('QQ Plot of Residuals - Log Model (Test Set)')
ax.get_lines()[0].set(markersize=4, alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_log_qq.png'), dpi=DPI)
plt.close()
print('Saved fig_log_qq.png')

print(f'\nAll 8 figures saved to {FIGDIR}/')
