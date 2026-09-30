"""权重记录读取: nvngx_dlssnr.dll 里抽出的 WEIGHTS_HT.bin + 记录表 weights_map.json (都在仓库根目录)。
记录按名字索引 (如 'block23.layer2.layer')，内容是 kernel 直接读取的原始字节。"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class Records:
    def __init__(self, root=ROOT):
        self.path = os.path.join(root, "WEIGHTS_HT.bin")
        self.recs = {r["name"]: r for r in json.load(open(os.path.join(root, "weights_map.json")))}

    def __getitem__(self, name):
        r = self.recs[name]
        with open(self.path, "rb") as f:
            f.seek(r["off"])
            return f.read(r["C"])

    def __contains__(self, name):
        return name in self.recs
