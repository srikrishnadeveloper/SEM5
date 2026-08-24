import pymupdf
doc = pymupdf.open(r'C:\Users\srik2\Downloads\EXP5-Consistent Hashing (1).pdf')
t = ''
for p in doc: t += p.get_text()
checks = [
    'Deploy three independent MongoDB instances',
    'Distributed Data Routing',
    'Application-Level Consistent Hashing',
    'modulo-based hashing',
    'Spring Boot application',
    'virtual nodes'
]
for c in checks:
    print(f'Contains: {c in t}')
PYEOF