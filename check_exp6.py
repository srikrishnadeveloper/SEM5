import pymupdf
doc = pymupdf.open(r'C:\Users\srik2\Downloads\EXP 6-Rate Limiter.pdf')
for p in doc:
    t = p.get_text().strip()
    if t:
        print(f'--- PAGE {p.number+1} ---')
        print(t[:2000])
        print('...')
PYEOF