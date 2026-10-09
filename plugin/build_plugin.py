"""打包 DLSS5-RTX30 插件: OptiScaler DLSSNR 发布版 + sm_86 移植版 nvngx_dlssnr.dll + 预设配置 + 安装/卸载脚本。

python plugin/build_plugin.py [--opti _dl/opti_020] [--dll sm86_port/nvngx_dlssnr.dll]
产物: plugin/dist/DLSS5-RTX30/
  payload/           原样拷进游戏目录的文件 (OptiScaler.dll 安装时再改名为代理 dll)
  DLSS5Manager.exe   图形界面管理工具 (扫描游戏库、检测、安装/卸载、调参数；源码 plugin/manager/)
  dlss5.exe          同一工具的命令行版
  install.ps1        单游戏安装脚本 (旧版，保留)
  使用说明.txt
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

# 安装时再按显卡覆盖 WorkingScale；这里只写与显卡无关的默认值
INI_OVERRIDES = {
    "DlssNr": {
        "Enabled": "true",       # 装上即开 (原版默认关)
        "AutoCapture": "false",  # 不在游戏目录里写调试截图
        "WorkingScale": "0.5",   # 占位，install.ps1 按显卡改写
    },
}


def patch_ini(text, overrides):
    """只改指定节里的 key=value，保留注释与其余内容；key 不存在则追加到节末尾。"""
    out, section, seen = [], None, set()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*\[(\w+)\]\s*$", line)
        if m:
            if section in overrides:
                for k, v in overrides[section].items():
                    if (section, k) not in seen:
                        out.append(f"{k}={v}")
            section = m.group(1)
        kv = re.match(r"(\w+)\s*=", line)
        if section in overrides and kv and kv.group(1) in overrides[section]:
            k = kv.group(1)
            out.append(f"{k}={overrides[section][k]}")
            seen.add((section, k))
            continue
        out.append(line)
    if section in overrides:
        for k, v in overrides[section].items():
            if (section, k) not in seen:
                out.append(f"{k}={v}")
    return "\r\n".join(out) + "\r\n"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opti", default=os.path.join(ROOT, "_dl", "opti_020"))
    ap.add_argument("--dll", default=os.path.join(ROOT, "sm86_port", "nvngx_dlssnr.dll"))
    ap.add_argument("--out", default=os.path.join(HERE, "dist", "DLSS5-RTX30"))
    a = ap.parse_args()

    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    payload = os.path.join(a.out, "payload")
    shutil.copytree(a.opti, payload, ignore=shutil.ignore_patterns(
        "setup_windows.bat", "setup_linux.sh", "!! EXTRACT ALL FILES TO GAME FOLDER !!"))

    ini = os.path.join(payload, "OptiScaler.ini")
    with open(ini, encoding="utf-8", errors="surrogateescape") as f:
        text = f.read()
    with open(ini, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write(patch_ini(text, INI_OVERRIDES))

    shutil.copy2(a.dll, os.path.join(payload, "nvngx_dlssnr.dll"))

    assert len(re.findall(r"(?m)^WorkingScale=", open(ini, encoding="utf-8", errors="surrogateescape").read())) == 1

    # .ps1 / .txt 带 BOM: Windows PowerShell 5.1 把无 BOM 的 UTF-8 当 ANSI 读，中文会乱码
    for src, dst in (("install.ps1", "install.ps1"), ("使用说明.txt", "使用说明.txt")):
        with open(os.path.join(HERE, "src", src), encoding="utf-8") as f:
            text = f.read()
        with open(os.path.join(a.out, dst), "w", encoding="utf-8-sig", newline="\r\n") as f:
            f.write(text)
    for src, dst in (("install.bat", "安装.bat"), ("uninstall.bat", "卸载.bat")):
        shutil.copy2(os.path.join(HERE, "src", src), os.path.join(a.out, dst))

    # 管理工具 (.NET Framework 4.8，VS2022 的 csc 编译)
    subprocess.run(["cmd", "/c", os.path.join(HERE, "manager", "build.bat")], check=True, stdout=subprocess.DEVNULL)
    for exe in ("DLSS5Manager.exe", "dlss5.exe"):
        shutil.copy2(os.path.join(HERE, "manager", "bin", exe), os.path.join(a.out, exe))

    files = []
    for dirpath, _, names in os.walk(payload):
        for n in names:
            p = os.path.join(dirpath, n)
            files.append({"path": os.path.relpath(p, payload).replace("\\", "/"), "size": os.path.getsize(p)})
    info = {
        "package": "DLSS5-RTX30",
        "optiscaler": os.path.basename(os.path.normpath(a.opti)),
        "dlssnr_sha256": sha256(a.dll),
        "files": sorted(files, key=lambda x: x["path"]),
    }
    with open(os.path.join(a.out, "package.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    total = sum(x["size"] for x in files)
    print(f"打包完成 {a.out}  {len(files)} 个文件  {total / 2**20:.0f} MiB  dlssnr {info['dlssnr_sha256'][:16]}…")


if __name__ == "__main__":
    main()
