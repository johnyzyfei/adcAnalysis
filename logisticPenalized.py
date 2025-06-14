import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample

# === Load data ===
file_path = r"C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\additional work sheet June 12, 2025.xlsx"
output_dir = r"C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\results_output"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
df.columns = df.columns.str.strip()

# === Define variables ===
predictor = "ADC diff % with min"
outcome = "Responders/Non R"
confounders = ["Age", "Gender", "HPV", "HIV"]
label = predictor

# === Clean and prepare data ===
data = df[[predictor, outcome] + confounders].dropna()
data[predictor] = data[predictor].clip(0, 100)
X = data[[predictor] + confounders]
y = data[outcome]

# === Fit penalized logistic regression ===
pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', solver='liblinear'))
pipe.fit(X, y)
model = pipe.named_steps['logisticregression']

# === Odds Ratios and Coefficients ===
coefs = np.concatenate([[model.intercept_[0]], model.coef_[0]])
odds_ratios = np.exp(coefs)
variables = ['Intercept'] + list(X.columns)

# === Bootstrap for 95% CI ===
boot_or = []
for _ in range(1000):
    Xr, yr = resample(X, y)
    bs_pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', solver='liblinear'))
    bs_pipe.fit(Xr, yr)
    bs_model = bs_pipe.named_steps['logisticregression']
    bs_coefs = np.concatenate([[bs_model.intercept_[0]], bs_model.coef_[0]])
    boot_or.append(np.exp(bs_coefs))

boot_or = np.array(boot_or)
ci_lower = np.percentile(boot_or, 2.5, axis=0)
ci_upper = np.percentile(boot_or, 97.5, axis=0)

# === Model evaluation: AUC, sensitivity, specificity ===
probs = pipe.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, probs)
roc_auc = roc_auc_score(y, probs)
best_thresh = thresholds[np.argmax(tpr - fpr)]
y_pred = (probs >= best_thresh).astype(int)

TP = ((y == 1) & (y_pred == 1)).sum()
TN = ((y == 0) & (y_pred == 0)).sum()
FP = ((y == 0) & (y_pred == 1)).sum()
FN = ((y == 1) & (y_pred == 0)).sum()

sensitivity = TP / (TP + FN) if TP + FN > 0 else np.nan
specificity = TN / (TN + FP) if TN + FP > 0 else np.nan

# === Formatting helper ===
def fmt(x):
    if pd.isna(x): return ""
    if isinstance(x, float): return f"{x:.3f}"
    return str(x)

# === Build summary DataFrame ===
summary = pd.DataFrame({
    "Variable": variables,
    "OR": [fmt(v) for v in odds_ratios],
    "2.5% CI": [fmt(v) for v in ci_lower],
    "97.5% CI": [fmt(v) for v in ci_upper],
    "p-value": [""] * len(variables)
})
summary.loc[len(summary)] = ["AUC", fmt(roc_auc), "", "", ""]
summary.loc[len(summary)] = ["Sensitivity", fmt(sensitivity), "", "", ""]
summary.loc[len(summary)] = ["Specificity", fmt(specificity), "", "", ""]

# === Save CSV ===
safe_label = label.lower().replace(" ", "_").replace("%", "pct")
csv_path = os.path.join(output_dir, f"logit_penalized_{safe_label}.csv")
summary.to_csv(csv_path, index=False)

# === Save summary table as PNG (with enough height) ===
fig_height = len(summary) * 0.6 + 2
fig, ax = plt.subplots(figsize=(8, fig_height))
ax.axis('off')
tbl = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.title(f"Penalized Logistic Regression – {label}", pad=20)
plt.tight_layout()
png_path = os.path.join(output_dir, f"logit_penalized_{safe_label}.png")
plt.savefig(png_path, dpi=300)
plt.close()

# === Save ROC Curve ===
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve – {label}")
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_path = os.path.join(output_dir, f"roc_penalized_{safe_label}.png")
plt.savefig(roc_path, dpi=300)
plt.close()

print("✅ All outputs saved:")
print(f"📊 Table PNG: {png_path}")
print(f"📈 ROC PNG:   {roc_path}")
print(f"📄 CSV:       {csv_path}")
