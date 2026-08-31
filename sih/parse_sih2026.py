import pandas as pd

html_path = r"C:\Users\srik2\Desktop\College\sih\sih2026PS_raw.html"
out_csv = r"C:\Users\srik2\Desktop\College\sih\sih2026PS_table.csv"

print("Parsing tables...")
tables = pd.read_html(html_path)
print(f"Found {len(tables)} tables")

for i, t in enumerate(tables):
    print(f"  Table {i}: {t.shape}")

largest = max(tables, key=lambda x: x.shape[0] * x.shape[1])
print(f"Largest table: {largest.shape}")
largest.to_csv(out_csv, index=False)
print(f"Saved to {out_csv}")
