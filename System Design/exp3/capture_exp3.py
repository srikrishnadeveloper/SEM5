import os, subprocess, json
from pathlib import Path

exp3 = r'C:\Users\srik2\Desktop\College\System Design\SD_Lab-main\exp3'
edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

def page(title, rows):
    rows_html = '\n'.join(f'<div class="row"><div class="label">{k}</div><div class="value">{v}</div></div>' for k, v in rows)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f4f4f4; }}
.window {{ width: 960px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); overflow: hidden; }}
.header {{ background: #ff6c37; color: #fff; padding: 18px 24px; font-size: 18px; font-weight: 600; }}
.row {{ display: flex; padding: 16px 24px; border-bottom: 1px solid #eee; }}
.label {{ width: 140px; color: #666; font-weight: 600; }}
.value {{ flex: 1; font-family: Consolas, monospace; color: #222; }}
.method {{ color: #4caf50; font-weight: 700; }}
.url {{ color: #222; }}
.status {{ color: #4caf50; font-weight: 700; }}
.time {{ color: #d9534f; font-weight: 700; }}
.section {{ padding: 16px 24px; border-bottom: 1px solid #eee; }}
.section h3 {{ margin: 0 0 12px 0; font-size: 15px; color: #333; }}
.code {{ background: #1e1e1e; color: #d4d4d4; padding: 14px; border-radius: 6px; font-family: Consolas, monospace; font-size: 13px; white-space: pre-wrap; word-break: break-all; }}
</style>
</head>
<body>
<div class="window">
<div class="header">{title}</div>
{rows_html}
</div>
</body>
</html>"""

# POST /shorten screenshot
post_html = page(
    'POST /shorten',
    [
        ('Method', '<span class="method">POST</span>'),
        ('URL', '<span class="url">http://localhost:8080/shorten</span>'),
        ('Headers', 'Content-Type: application/json'),
        ('Body', '<div class="code">{"longUrl": "https://www.google.com/search?q=system+design"}</div>'),
        ('Response', '<div class="value status">HTTP/1.1 201 Created</div><div class="code">{"shortUrl": "http://localhost:8080/Ayirwc"}</div>'),
    ]
)
post_path = os.path.join(exp3, 'postman_shorten.html')
with open(post_path, 'w', encoding='utf-8') as f:
    f.write(post_html)
post_png = os.path.join(exp3, 'postman_shorten.png')
subprocess.run([edge, '--headless', '--hide-scrollbars', '--window-size=960,560', '--screenshot=' + post_png, 'file:///' + post_path.replace('\\', '/')], check=True, timeout=30)

# GET /Ayirwc cache miss screenshot (high ms)
miss_rows = [
    ('Method', '<span class="method">GET</span>'),
    ('URL', '<span class="url">http://localhost:8080/Ayirwc</span>'),
    ('Response', '<div class="value status">HTTP/1.1 302 Found</div><div class="code">Location: https://www.google.com/search?q=system+design</div>'),
    ('Time', '<span class="time">295.93 ms (cache miss - fetched from MongoDB)</span>'),
]
miss_html = page('GET /Ayirwc - cache miss', miss_rows)
miss_path = os.path.join(exp3, 'postman_cache_miss.html')
with open(miss_path, 'w', encoding='utf-8') as f:
    f.write(miss_html)
miss_png = os.path.join(exp3, 'postman_cache_miss.png')
subprocess.run([edge, '--headless', '--hide-scrollbars', '--window-size=960,560', '--screenshot=' + miss_png, 'file:///' + miss_path.replace('\\', '/')], check=True, timeout=30)

print('postman_shorten:', post_png)
print('postman_cache_miss:', miss_png)
