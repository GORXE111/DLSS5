"""anatomy2.py 用的公开测试画面: 从维基共享资源 (Wikimedia Commons) 下载到 _dl/commons/ (不入库)。

    python research/fetch_commons.py            # 按 commons_manifest.json 下载同一批 68 张 (人物 27、易混淆物 9、场景 26、CG 6)
    python research/fetch_commons.py --search   # 按下面的关键词重新搜索 (结果随时间变化，会得到另一批图)，并改写清单

只取横幅 (宽 >= 1.2 x 高、宽 >= 1600) 的 jpeg / png，下载 1920 宽的缩略图，存成 <类别>_<序号>.jpg。
commons_manifest.json 只记文件名、类别、Commons 标题、许可与页面地址，不含图片；图片各有各的许可 (CC BY-SA、CC0、公有领域等)，
只在本地做研究用，不随仓库分发。
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "_dl", "commons")
MANIFEST = os.path.join(HERE, "commons_manifest.json")
UA = {"User-Agent": "DLSS5-research/0.1 (local image study)"}
# (类别, 搜索词, 张数)。people: 不同肤色 / 年龄 / 远近的人；decoy: 颜色或质感像皮肤但不是人的东西
SETS = [
    ("people", 'incategory:"Featured_pictures_of_people" portrait', 10),
    ("people", 'incategory:"Quality_images_of_people" portrait smiling', 8),
    ("people", 'incategory:"Quality_images_of_people" group people street', 6),
    ("people", 'incategory:"Quality_images_of_people" african portrait', 6),
    ("people", 'incategory:"Quality_images_of_people" indian portrait woman', 5),
    ("people", 'incategory:"Quality_images_of_people" elderly man', 4),
    ("people", 'incategory:"Quality_images_of_people" child', 4),
    ("people", 'incategory:"Quality_images_of_people" hands', 3),
    ("decoy", 'incategory:"Featured_pictures_of_mammals" lion', 4),
    ("decoy", 'incategory:"Quality_images_of_sculptures" marble statue', 4),
    ("decoy", 'incategory:"Quality_images" sand dunes desert', 3),
    ("decoy", 'incategory:"Quality_images" wooden furniture', 3),
    ("decoy", 'incategory:"Quality_images" bread bakery', 2),
    ("scene", 'incategory:"Featured_pictures_of_landscapes" mountains', 5),
    ("scene", 'incategory:"Quality_images" night city street lights', 5),
    ("scene", 'incategory:"Quality_images" church interior', 4),
    ("scene", 'incategory:"Quality_images" forest', 4),
    ("scene", 'incategory:"Quality_images" snow winter', 3),
    ("scene", 'incategory:"Quality_images" sunset sea', 3),
    ("scene", 'incategory:"Quality_images" underwater', 2),
    ("cg", 'Blender render 3D scene filetype:bitmap', 6),
    ("cg", 'computer generated imagery render interior filetype:bitmap', 4),
]


def api(params):
    q = urllib.parse.urlencode(dict(params, action="query", format="json"))
    return json.load(urllib.request.urlopen(urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + q, headers=UA), timeout=60))


def fetch(url, path):
    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
    open(path, "wb").write(data)


def from_manifest():
    """清单里的每张图: 按标题查 1920 宽缩略图的地址再下载 (已有的跳过)"""
    man = json.load(open(MANIFEST, encoding="utf-8"))
    todo = [m for m in man if not os.path.exists(os.path.join(OUT, m["file"]))]
    print(f"清单 {len(man)} 张，待下载 {len(todo)} 张")
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        d = api({"titles": "|".join(m["title"] for m in batch), "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 1920})
        urls = {p["title"]: p["imageinfo"][0]["thumburl"] for p in d["query"]["pages"].values() if "imageinfo" in p}
        norm = {n["from"]: n["to"] for n in d["query"].get("normalized", [])}
        for m in batch:
            u = urls.get(norm.get(m["title"], m["title"]))
            if u is None:
                print("找不到 (可能已被删除或改名):", m["title"])
                continue
            try:
                fetch(u, os.path.join(OUT, m["file"]))
                print(m["file"], m["title"][:70])
            except Exception as e:      # noqa: BLE001
                print("失败", m["file"], e)
            time.sleep(0.5)


def search():
    man, seen, count = [], set(), {}
    for tag, query, n in SETS:
        d = api({"generator": "search", "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": 40, "prop": "imageinfo",
                 "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1920})
        got = 0
        for p in sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
            ii = p["imageinfo"][0]
            if p["title"] in seen or ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if ii["width"] < 1.2 * ii["height"] or ii["width"] < 1600:
                continue
            fn = f"{tag}_{count.get(tag, 0) + 1:02d}.jpg"
            try:
                fetch(ii["thumburl"], os.path.join(OUT, fn))
            except Exception as e:      # noqa: BLE001
                print("跳过", p["title"], e)
                continue
            count[tag] = count.get(tag, 0) + 1
            man.append({"file": fn, "tag": tag, "title": p["title"], "url": ii["descriptionurl"],
                        "license": ii["extmetadata"].get("LicenseShortName", {}).get("value", "")})
            seen.add(p["title"])
            got += 1
            print(fn, p["title"][:70])
            time.sleep(0.5)
            if got >= n:
                break
    json.dump(man, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(count)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="重新搜索并改写清单 (得到的不是同一批图)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    search() if a.search else from_manifest()
