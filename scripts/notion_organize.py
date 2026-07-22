import json
import time
import urllib.request
import urllib.error

ENV_PATH = r"C:\Users\srik2\Desktop\College\.env"
TARGET_PAGE = "364d2187-71ce-8021-aef6-c0bb9f2f8510"


def get_token():
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("notion_space1"):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("notion_space1 not found in .env")


TOKEN = get_token()
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def request(method, url, body=None, retries=6):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            print("HTTP ERROR", e.code, e.read().decode())
            raise
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise last_err


def mention_bullet(page_id):
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "mention", "mention": {"type": "page", "page": {"id": page_id}}}
            ]
        },
    }


def heading3(text):
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def heading2(text):
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


ACADEMICS = {
    "Database Management Systems": ["361d2187-71ce-805c-83e6-d59f978d2ef4", "35dd2187-71ce-8010-ba51-c7a5abd4960a"],
    "Internet Programming": ["35bd2187-71ce-80a0-aaf1-c331a336fde2"],
    "Artificial Intelligence": ["35ad2187-71ce-802d-9d59-d98c8a8eeffb", "314d2187-71ce-80c2-9d82-dcc9a8fc8902"],
    "Algorithms (DAA)": ["356d2187-71ce-8094-95ad-fa594f97c713", "337d2187-71ce-80b9-9387-d4a650167c2f", "331d2187-71ce-806e-b424-f6425b9fe6f5", "31fd2187-71ce-80af-9bbf-c980508fed8c"],
    "Indian Constitution": ["357d2187-71ce-80af-b7ed-e7a688d520c0"],
    "Mathematics": ["2eed2187-71ce-802a-9bad-d67364c65beb", "31fd2187-71ce-8075-81ad-ed9045b62395"],
    "Image Processing": ["30cd2187-71ce-8089-a6e1-ea553b758124"],
}

PROJECTS = {
    "E-Commerce": ["33fd2187-71ce-80cd-933b-da75612c1623", "2b9d2187-71ce-801e-8d71-cadcf7e5589d", "2d0d2187-71ce-809d-827a-dc83c0e8f0fa"],
    "DBMS Project": ["331d2187-71ce-8023-9ed0-f33f3dcf6472"],
    "React": ["31fd2187-71ce-80d8-8e1c-d5658dc7b4d0"],
    "Docker": ["314d2187-71ce-807f-8a87-c28533d11a3b"],
    "JavaScript": ["30cd2187-71ce-8003-9df8-fb56c7617312"],
    "WhatsApp Automation": ["27ed2187-71ce-80c9-bf45-fd3c78c6b85c", "318d2187-71ce-8043-a263-fc114a586d43"],
}


def build_blocks():
    blocks = [{"object": "block", "type": "divider", "divider": {}}]
    blocks.append(heading2("\U0001F4DA Academics"))
    for subject, ids in ACADEMICS.items():
        blocks.append(heading3(subject))
        for pid in ids:
            blocks.append(mention_bullet(pid))
    blocks.append(heading2("\U0001F6E0️ Projects"))
    for proj, ids in PROJECTS.items():
        blocks.append(heading3(proj))
        for pid in ids:
            blocks.append(mention_bullet(pid))
    return blocks


def main():
    before = request("GET", f"https://api.notion.com/v1/blocks/{TARGET_PAGE}/children?page_size=100")
    before_count = len(before.get("results", []))
    print("Existing block count before:", before_count)

    blocks = build_blocks()
    print("Blocks to append:", len(blocks))

    result = request(
        "PATCH",
        f"https://api.notion.com/v1/blocks/{TARGET_PAGE}/children",
        {"children": blocks},
    )
    appended = len(result.get("results", []))
    print("Blocks appended per API response:", appended)

    after = request("GET", f"https://api.notion.com/v1/blocks/{TARGET_PAGE}/children?page_size=100")
    after_count = len(after.get("results", []))
    print("Existing block count after:", after_count)

    if after_count - before_count == len(blocks):
        print("SUCCESS: block count matches expected addition.")
    else:
        print(f"WARNING: expected +{len(blocks)}, got +{after_count - before_count}")


if __name__ == "__main__":
    main()
