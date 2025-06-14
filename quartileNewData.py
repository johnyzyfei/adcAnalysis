import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r"C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\additional work sheet June 12, 2025.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")

# Clean column names
df.columns = df.columns.str.strip()

# Define your predictor and outcome columns
predictor = "ADC diff % with min"
outcome = "Responders/Non R"

# Subset and remove missing values
data = df[[predictor, outcome]].dropna()

# Create quartiles
data["Quartile"] = pd.qcut(data[predictor], q=4)

# Cross-tabulation
ct = pd.crosstab(data["Quartile"], data[outcome])
print(ct)

# Plot
ct.plot(kind="bar", stacked=True, colormap="viridis", figsize=(8, 6))
plt.title("Responders vs Non-Responders by ADC diff % with min Quartile")
plt.xlabel("ADC diff % with min Quartile")
plt.ylabel("Count")
plt.legend(title="Response")
plt.tight_layout()

# Save the plot
plt.savefig(r"C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\adc_diff_min_quartile_plot.png", dpi=300)

# Show it
plt.show()

