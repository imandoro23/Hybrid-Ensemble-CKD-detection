import pandas as pd

# Read just the category column to see what's available
df = pd.read_parquet(
    r"C:\Users\imand\Downloads\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0\clif_vitals.parquet",
    columns=['vital_category','vital_name']
)
print("Vital categories:")
print(df['vital_category'].value_counts().to_string())
print("\nVital names (top 30):")
print(df['vital_name'].value_counts().head(30).to_string())