import re
import csv
from lxml import html

html_path = r"C:\Users\srik2\Desktop\College\sih\sih2026PS_raw.html"
out_csv = r"C:\Users\srik2\Desktop\College\sih\sih2026_problem_statements.csv"

with open(html_path, "r", encoding="utf-8") as f:
    raw = f.read()

# Replace <br> tags with newlines so descriptions stay readable
raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)

tree = html.fromstring(raw)

table = tree.xpath('//table[@id="dataTablePS"]')[0]
rows = table.xpath('.//tbody/tr')

main_headers = [
    "S.No.",
    "Organization",
    "Problem Statement Title",
    "Category",
    "PS Number",
    "Submitted Idea(s) Count",
    "Theme",
    "Deadline for Idea Submission",
]

detail_keys = [
    "Problem Statement ID",
    "Problem Statement Title",
    "Description",
    "Organization",
    "Department",
    "Category",
    "Theme",
    "Youtube Link",
    "Dataset Link",
    "Contact info",
]

records = []

for row in rows:
    tds = row.xpath('./td')
    if len(tds) < 8:
        continue

    s_no = tds[0].text_content().strip()
    if not s_no.isdigit():
        continue

    org = tds[1].text_content().strip()
    title = ""
    a = tds[2].xpath('.//a')
    if a:
        title = a[0].text_content().strip()
    category = tds[3].text_content().strip()
    ps_num = tds[4].text_content().strip()
    submitted = tds[5].text_content().strip()
    theme = tds[6].text_content().strip()
    deadline = tds[7].text_content().strip()

    record = {k: "" for k in main_headers}
    record.update({
        "S.No.": s_no,
        "Organization": org,
        "Problem Statement Title": title,
        "Category": category,
        "PS Number": ps_num,
        "Submitted Idea(s) Count": submitted,
        "Theme": theme,
        "Deadline for Idea Submission": deadline,
    })

    # extract detail table from the title cell
    settings = tds[2].xpath('.//table[@id="settings"]')
    if settings:
        for tr in settings[0].xpath('.//tr'):
            ths = tr.xpath('./th')
            tds2 = tr.xpath('./td')
            if ths and tds2:
                key = ths[0].text_content().strip()
                val = tds2[0].text_content().strip()
                # collapse multiple newlines to single spaces for non-description fields
                if key != "Description":
                    val = re.sub(r"\s+", " ", val)
                else:
                    # collapse excessive blank lines but keep paragraphs
                    val = re.sub(r"\n{3,}", "\n\n", val)
                    val = val.strip()
                # skip keys already covered by main headers; keep others
                if key in detail_keys and key not in main_headers:
                    record[key] = val

    records.append(record)

# build fieldnames: main headers first, then details not already in main headers
extra = [k for k in detail_keys if k not in main_headers]
fieldnames = main_headers + extra

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in records:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

print(f"Wrote {len(records)} problem statements to {out_csv}")
