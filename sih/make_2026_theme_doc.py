import pandas as pd

csv_path = r"C:\Users\srik2\Desktop\College\sih\sih2026_problem_statements.csv"
out_md = r"C:\Users\srik2\Desktop\College\sih\06_2026_ps_themes.md"

df = pd.read_csv(csv_path)

software = (df["Category"].str.lower() == "software").sum()
hardware = (df["Category"].str.lower() == "hardware").sum()

lines = ["# SIH 2026 Problem Statements — Downloaded Summary\n"]
lines.append("Source: https://sih.gov.in/sih2026PS\n")
lines.append(f"- **Total problem statements:** {len(df)}\n")
lines.append(f"- **Software:** {software}\n")
lines.append(f"- **Hardware:** {hardware}\n")
lines.append(f"- **Idea submission deadline:** {df['Deadline for Idea Submission'].iloc[0]}\n\n")

lines.append("## Themes and counts\n\n")
theme_counts = df["Theme"].value_counts().sort_index()
for theme, count in theme_counts.items():
    lines.append(f"- {theme}: {count}\n")

lines.append("\n## Sample problem statements\n\n")
lines.append("| S.No. | Organization | Title | Category | PS Number | Theme |\n")
lines.append("|---|---|---|---|---|---|\n")

for _, row in df.head(15).iterrows():
    title = str(row["Problem Statement Title"]).replace("|", "\\|")
    org = str(row["Organization"]).replace("|", "\\|")
    lines.append(f"| {row['S.No.']} | {org} | {title} | {row['Category']} | {row['PS Number']} | {row['Theme']} |\n")

lines.append("\n## Full download files\n\n")
lines.append("- `sih2026_problem_statements.csv` — full list with descriptions\n")
lines.append("- `sih2026_problem_statements.xlsx` — full list in Excel\n")
lines.append("- `sih2026_summary.md` — theme and category counts\n")

with open(out_md, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Wrote {out_md}")
