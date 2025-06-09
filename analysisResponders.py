import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# For displaying results in table
# -----------------------------------------------------------------------------
def fmt_num(x, small=1e-3, large=1e3, prec=3):
    """
    Format floats for a results table:
      - x <= 0 or x < small    -> '<{small:.{prec}f}'
      - x > large               -> '>{large:.{prec}f}'
      - NaN                     -> ''
      - else                    -> '{x:.{prec}f}'
    """
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
# -----------------------------------------------------------------------------

# Paths & I/O setup
file_path  = r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\Quantitative ADC in Anal ca  Data sheet APRIL May 28, 2025 version 2.xlsx'
output_dir = r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\results_output'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_excel(file_path, sheet_name='analysis sheet', engine='openpyxl')

# Define outcome, confounders and predictors
outcome     = 'Responders/Non R'
confounders = ['Age', 'Gender', 'HPV', 'HIV']
predictors = {
    'Pre ADC Min'           : 'Pre ADC Min',
    'Pre ADC Mean'          : 'Pre ADC Mean ',
    'ADC diff with min'     : 'ADC diff with min',
    'ADC diff with mean'    : 'ADC diff with mean ',
    'ADC diff % with min'   : 'ADC diff % with min ',
    'ADC diff % with mean'  : 'ADC diff % with mean '
}

for label, predictor in predictors.items():
    # Skip "ADC diff % with min"
    if label == 'ADC diff % with min':
        print(f"\n=== {label} skipped ===")
        continue

    print(f"\n=== {label} ===")

    # 1) subset & clean
    cols = [outcome, predictor] + confounders
    data = df[cols].apply(pd.to_numeric, errors='coerce').dropna()

    # 2) clip percents into [0,100]
    if '%' in label:
        data[predictor] = data[predictor].clip(0, 100)

    y = data[outcome]

    # Always multivariable fit
    X = data[[predictor] + confounders]
    method_tag = 'multivariable'

    # add constant and fit
    Xc  = sm.add_constant(X)
    fit = sm.Logit(y, Xc).fit(disp=False)
    pred_probs = fit.predict(Xc)

    # extract ORs, CIs, p-values
    ORs   = np.exp(fit.params)
    ci    = fit.conf_int()
    ci.columns = ['2.5% CI','97.5% CI']
    ci    = np.exp(ci)
    pvals = fit.pvalues

    # build summary dataframe
    summary = pd.DataFrame({
        'Variable': ORs.index,
        'OR'      : ORs.values,
        '2.5% CI' : ci['2.5% CI'].values,
        '97.5% CI': ci['97.5% CI'].values,
        'p-value' : pvals.values
    })

    # compute AUC and append as last row
    fpr, tpr, _ = roc_curve(y, pred_probs)
    roc_auc     = auc(fpr, tpr)
    summary.loc[len(summary)] = ['AUC', roc_auc, np.nan, np.nan, np.nan]

    # formatting
    for col in ['OR','2.5% CI','97.5% CI','p-value']:
        summary[col] = summary[col].apply(fmt_num)

    # saving CSV
    safe = label.lower().replace(' ','_').replace('%','pct')
    summary.to_csv(os.path.join(output_dir, f'logistic_results_{safe}.csv'), index=False)

    # plotting ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC – {label} ({method_tag})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_{safe}.png'), dpi=300)
    plt.close()

    # save table figure
    fig, ax = plt.subplots(figsize=(8, len(summary)*0.5 + 1))
    ax.axis('off')
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2,1.2)
    plt.title(f'Logistic Regression – {label} ({method_tag})', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'table_{safe}.png'), dpi=300)
    plt.close()

    print(f"Outputs saved for {label}  (method={method_tag}, AUC={roc_auc:.3f})")