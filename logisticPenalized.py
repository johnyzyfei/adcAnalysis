import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

# For displaying results in table
# -----------------------------------------------------------------------------
def fmt_num(x, small=1e-3, large=1e3, prec=3):
    try:
        if pd.isna(x):
            return ""
        if x <= 0 or x < small:
            return f"<{small:.{prec}f}"
        if x > large:
            return f">{large:.{prec}f}"
        return f"{x:.{prec}f}"
    except:
        return str(x)
# ---------------------------------------------------------------------

# Load and clean data
file_path = r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\Quantitative ADC in Anal ca  Data sheet APRIL May 28, 2025 version 2.xlsx'
df = pd.read_excel(file_path, sheet_name='analysis sheet', engine='openpyxl')

outcome = 'Responders/Non R'
exposure = 'ADC diff % with min '
confounders = ['Age', 'Gender', 'HPV', 'HIV']
df = df[[outcome, exposure] + confounders].apply(pd.to_numeric, errors='coerce').dropna()
df[exposure] = df[exposure].clip(0, 100)

y = df[outcome]
X = sm.add_constant(df[[exposure] + confounders])

# Fit Ridge logistic regression
model = sm.Logit(y, X)
result = model.fit_regularized(alpha=0.1, L1_wt=0, maxiter=1000)

# Extract odds ratios
ORs = np.exp(result.params)

# Bootstrap for 95% CI
B = 500
boot_coefs = []
np.random.seed(42)
for _ in range(B):
    sample_idx = np.random.choice(len(df), size=len(df), replace=True)
    Xb = X.iloc[sample_idx]
    yb = y.iloc[sample_idx]
    try:
        boot_fit = sm.Logit(yb, Xb).fit_regularized(alpha=0.1, L1_wt=0, disp=0)
        boot_coefs.append(boot_fit.params.values)
    except:
        continue

boot_coefs = np.array(boot_coefs)
ci_lower = np.exp(np.percentile(boot_coefs, 2.5, axis=0))
ci_upper = np.exp(np.percentile(boot_coefs, 97.5, axis=0))

# Two-tailed p-values from bootstrap distribution
pvals = []
for j, val in enumerate(result.params):
    null_dist = boot_coefs[:, j]
    p = np.mean(np.abs(null_dist) >= np.abs(val))
    pvals.append(p)

# Compute AUC
probs = result.predict(X)
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

# Create summary DataFrame
summary = pd.DataFrame({
    'Variable': X.columns,
    'OR': ORs,
    '2.5% CI': ci_lower,
    '97.5% CI': ci_upper,
    'p-value': pvals
})
summary.loc[len(summary)] = ['AUC', roc_auc, np.nan, np.nan, np.nan]

# Format all numeric columns using fmt_num()
for col in ['OR', '2.5% CI', '97.5% CI', 'p-value']:
    summary[col] = summary[col].apply(fmt_num)

# Save summary table as image
fig, ax = plt.subplots(figsize=(9, len(summary) * 0.5 + 1))
ax.axis('off')
tbl = ax.table(cellText=summary.values,
               colLabels=summary.columns,
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.title('Penalized Logistic Regression – ADC diff % with min', pad=20)
plt.tight_layout()
plt.savefig(r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\results_output\table_adc_diff_pct_with_min_penalized_full.png', dpi=300)
plt.close()

# Save ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – ADC diff % with min (Penalized)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\results_output\roc_adc_diff_pct_with_min_penalized.png', dpi=300)
plt.close()

print(summary)
