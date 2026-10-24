import pandas as pd

files = ['clif_patient.parquet',
         'clif_hospital_diagnosis.parquet',
         'clif_labs.parquet',
         'clif_hospitalization.parquet']

folder = r"C:\Users\imand\Downloads\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0\mimic-iv-ext-clif-mimic-iv-in-the-common-longitudinal-icu-data-format-clif-1.1.0"

for f in files:
    try:
        df = pd.read_parquet(f"{folder}\\{f}")
        print(f"\n{f}")
        print(f"  Shape:   {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample:  {df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"{f}: ERROR — {e}")