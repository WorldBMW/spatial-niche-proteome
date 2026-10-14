import pandas as pd

xl = pd.ExcelFile("results_vitl14_jbi/enrichment.xlsx")
print("sheets:", xl.sheet_names)
for s in xl.sheet_names:
    df = xl.parse(s)
    print(f"\n===== sheet: {s} (rows={len(df)}) =====")
    print("columns:", list(df.columns))
    print(df.head(30).to_string())