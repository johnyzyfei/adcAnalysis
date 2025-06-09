import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file )
file_path = r"C:\Users\johny\OneDrive\Desktop\Lab\adc_analysis\Quantitative ADC in Anal ca  Data sheet APRIL May 28, 2025 version 2.xlsx"
df = pd.read_excel(file_path, sheet_name="analysis sheet", engine="openpyxl")

# Strip column names 
df.columns = df.columns.str.strip()

# Check the correct column name
column_name = "ADC diff with mean"
outcome_col = "Responders/Non R"

# Subset data and remove rows with missing values
data = df[[column_name, outcome_col]].dropna()

# Create quartile bins for the predictor
data["Quartile"] = pd.qcut(data[column_name], q=4)

# Cross-tabulate the quartiles with the response
ct = pd.crosstab(data["Quartile"], data[outcome_col])
print("Cross-tabulation:\n", ct)

# Plottinh
ct.plot(kind="bar", stacked=True, colormap="viridis", figsize=(8, 6))
plt.title("Responders vs Non-Responders by ADC diff with mean Quartile")
plt.xlabel("ADC diff with mean Quartile")
plt.ylabel("Count")
plt.legend(title="Response")
plt.tight_layout()
plt.show()
