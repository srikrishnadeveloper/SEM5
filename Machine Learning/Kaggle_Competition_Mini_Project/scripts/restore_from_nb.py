import json
import ast
import re
from pathlib import Path

nb = json.load(open("notebooks/V8_1_Kaggle_Production.ipynb", encoding="utf-8"))
c4 = "".join(nb["cells"][3]["source"])
files = [
    "metrics/pq.py",
    "scripts/convert_coco_to_yolo.py",
    "v8_1/config.py",
    "v8_1/geometry.py",
    "v8_1/1_train_yolo.py",
    "v8_1/2_train_crop_refiner.py",
    "v8_1/3_infer_cascade.py",
]
for fn in files:
    pattern = rf'with open\("{re.escape(fn)}"[^\)]*\) as f:\s+f\.write\((.*?)\)\n'
    m = re.search(pattern, c4, re.DOTALL)
    if m:
        content = ast.literal_eval(m.group(1))
        p = Path(fn)
        # only restore if file is 0 bytes or doesn't exist
        if not p.exists() or p.stat().st_size == 0:
            p.write_text(content, encoding="utf-8")
            print(f"Restored {fn} ({len(content)} chars)")
        else:
            print(f"Kept existing {fn} ({p.stat().st_size} bytes)")
    else:
        print(f"Not found in notebook: {fn}")
