import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the second sheet from the Excel file
file_path = r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\Quantitative ADC in Anal ca  Data sheet APRIL May 28, 2025 version 2.xlsx'
df = pd.read_excel(file_path, sheet_name="analysis sheet", engine="openpyxl")

# Define the columns of interest
columns_of_interest = [
    'Pre ADC Min Whole group',
    'Pre ADC Min CR',
    'Pre ADC Mean  CR',
    'Pre ADC  MinPR',
    'Pre ADC Mean PR',
    'Post ADC min CR',
    'Post ADC Mean CR',
    'Post ADC minPR',
    'Post ADC Mean PR'
]

# Strip column names 
df.columns = df.columns.str.strip()

# Compute means
adc_means = df[columns_of_interest].mean(numeric_only=True).round(3)
summary_df = adc_means.reset_index()
summary_df.columns = ['Variable', 'Mean']

# Save as CSV
output_dir = r'C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\results_output'
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'adc_summary_means.csv')
summary_df.to_csv(csv_path, index=False)

# Save as PNG table
fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.5 + 1))
ax.axis('off')
table = ax.table(cellText=summary_df.values,
                 colLabels=summary_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Mean ADC Values by Group', pad=20)
plt.tight_layout()
png_path = os.path.join(output_dir, 'adc_summary_means.png')
plt.savefig(png_path, dpi=300)
plt.close()

print(f" Summary saved as:\n→ {csv_path}\n→ {png_path}")
