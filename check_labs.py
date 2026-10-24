import pandas as pd

labs = pd.read_parquet(
    r"C:\Users\imand\Downloads\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0\clif_labs.parquet",
    columns=['lab_category','lab_name']
)
print("Lab categories (all):")
print(labs['lab_category'].value_counts().head(50).to_string())
print("\nLab names containing 'hematocrit' or 'haematocrit':")
mask = labs['lab_name'].str.lower().str.contains('hematocrit|haematocrit', na=False)
print(labs[mask][['lab_category','lab_name']].drop_duplicates().to_string())
print("\nLab names containing 'rbc' or 'red blood':")
mask2 = labs['lab_name'].str.lower().str.contains('rbc|red blood', na=False)
print(labs[mask2][['lab_category','lab_name']].drop_duplicates().to_string())