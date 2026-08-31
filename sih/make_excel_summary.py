import pandas as pd

csv_path = r"C:\Users\srik2\Desktop\College\sih\sih2026_problem_statements.csv"
out_xlsx = r"C:\Users\srik2\Desktop\College\sih\sih2026_problem_statements.xlsx"
summary_md = r"C:\Users\srik2\Desktop\College\sih\sih2026_summary.md"

df = pd.read_csv(csv_path)

# Excel
df.to_excel(out_xlsx, index=False)

# Summary
software = (df["Category"].str.lower() == "software").sum()
hardware = (df["Category"].str.lower() == "hardware").sum()
total = len(df)

def to_markdown_table(frame):
    lines = []
    cols = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in [""] + cols) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for idx, row in frame.iterrows():
        cells = [str(idx)]
        for c in cols:
            cells.append(str(row[c]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# SIH 2026 Problem Statements Summary\n\n")
    f.write(f"- **Total problem statements:** {total}\n")
    f.write(f"- **Software:** {software}\n")
    f.write(f"- **Hardware:** {hardware}\n")
    f.write(f"- **Deadline:** {df['Deadline for Idea Submission'].iloc[0] if len(df) else 'N/A'}\n\n")

    f.write("## Counts by theme\n\n")
    theme_counts = df["Theme"].value_counts().sort_index()
    for theme, count in theme_counts.items():
        f.write(f"- {theme}: {count}\n")

    f.write("\n## Counts by category and theme\n\n")
    cat_theme = df.groupby(["Category", "Theme"]).size().unstack(fill_value=0)
    f.write(to_markdown_table(cat_theme))

print(f"Wrote Excel: {out_xlsx}")
print(f"Wrote summary: {summary_md}")
