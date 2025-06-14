import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
file_path = r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\Quantitative ADC in Anal ca  Data sheet APRIL May 28, 2025 version 2.xlsx'
df = pd.read_excel(file_path, sheet_name='analysis sheet', engine='openpyxl')

# Outcome and confounders for the second set
outcome = 'Response'
confounders = ['Age', 'Gender', 'HPV', 'HIV']

# Predictor columns
predictors = {
    'Pre ADC Min': 'Pre ADC Min',
    'Pre ADC Mean': 'Pre ADC Mean ',
    'ADC diff with min': 'ADC diff with min',
    'ADC diff with mean': 'ADC diff with mean ',
    'ADC diff % with min': 'ADC diff % with min ',
    'ADC diff % with mean': 'ADC diff % with mean '
}

# Output directory
output_dir = r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\results_output_set2'
os.makedirs(output_dir, exist_ok=True)

for label, predictor in predictors.items():
    print(f'\n--- Running analysis for: {label} ---\n')

    cols = [outcome, predictor] + confounders
    data = df[cols].apply(pd.to_numeric, errors='coerce').dropna()

    if '%' in label:
        data[predictor] = data[predictor].clip(lower=0, upper=100)

    print("Checking for singularity, variance and correlation...")
    print("Value distribution:")
    print(data[predictor].describe())
    print("Unique values:", data[predictor].nunique())
    print("\nCorrelation with confounders:")
    print(data[[predictor] + confounders].corr()[predictor])

    y = data[outcome]
    X = sm.add_constant(data[[predictor] + confounders])

    try:
        model = sm.Logit(y, X).fit(disp=0)

        # ORs and CIs
        odds_ratios = model.params.apply(lambda x: round(np.exp(x), 3))
        conf_int = model.conf_int()
        conf_int.columns = ['2.5% CI', '97.5% CI']
        conf_int['2.5% CI'] = conf_int['2.5% CI'].apply(lambda x: round(np.exp(x), 3))
        conf_int['97.5% CI'] = conf_int['97.5% CI'].apply(lambda x: round(np.exp(x), 3))

        # Build summary table
        summary_df = pd.DataFrame({
            'Variable': odds_ratios.index,
            'OR': odds_ratios.values,
            '2.5% CI': conf_int['2.5% CI'],
            '97.5% CI': conf_int['97.5% CI'],
            'p-value': model.pvalues.round(3).values
        })

        # AUC and ROC
        pred_probs = model.predict(X)
        fpr, tpr, thresholds = roc_curve(y, pred_probs)
        roc_auc = auc(fpr, tpr)

        # Optimal threshold (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        # Binary predictions
        y_pred = (pred_probs >= best_thresh).astype(int)

        # Confusion matrix components
        TP = np.sum((y == 1) & (y_pred == 1))
        TN = np.sum((y == 0) & (y_pred == 0))
        FP = np.sum((y == 0) & (y_pred == 1))
        FN = np.sum((y == 1) & (y_pred == 0))

        # Sensitivity & Specificity
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan

        # Append AUC, Sensitivity, Specificity
        summary_df.loc[len(summary_df)] = ['AUC', round(roc_auc, 3), '', '', '']
        summary_df.loc[len(summary_df)] = ['Sensitivity', round(sensitivity, 3), '', '', '']
        summary_df.loc[len(summary_df)] = ['Specificity', round(specificity, 3), '', '', '']

        # Save CSV
        safe_label = label.lower().replace(' ', '_').replace('%', 'pct')
        csv_path = os.path.join(output_dir, f'logistic_results_{safe_label}_set2_unpenalized.csv')
        summary_df.to_csv(csv_path, index=False)

        # Save ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC – {label} (Outcome: Response)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{safe_label}_set2_unpenalized.png'), dpi=300)
        plt.close()

        # Save summary table as PNG
        fig, ax = plt.subplots(figsize=(8, len(summary_df) * 0.5 + 1))
        ax.axis('off')
        table = ax.table(cellText=summary_df.values,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title(f'Logistic Regression Summary – {label}', fontsize=12, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'logistic_results_{safe_label}_set2_unpenalized.png'), dpi=300)
        plt.close()

        print(f' Completed: {label} | AUC = {roc_auc:.3f} | Sensitivity = {sensitivity:.3f} | Specificity = {specificity:.3f}')
    except Exception as e:
        print(f' Failed for {label} due to: {e}')
