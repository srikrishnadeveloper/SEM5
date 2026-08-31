import re
from lxml import html

path = r"C:\Users\srik2\Desktop\College\sih\sih2026PS_raw.html"
with open(path, 'r', encoding='utf-8') as f:
    raw = f.read()
raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
t = html.fromstring(raw)
table = t.xpath('//table[@id="dataTablePS"]')[0]
row = table.xpath('.//tbody/tr')[0]
tds = row.xpath('./td')
print('tds count', len(tds))
print('a tags count', len(tds[2].xpath('.//a')))
for i, a in enumerate(tds[2].xpath('.//a')[:3]):
    print(i, 'a text:', repr(a.text), 'content:', repr(a.text_content()))
