"""兜底模式在真实 3D DX11 程序里的测试: Unigine Superposition 1.1 (64 位，免费的基础版)。

    python superposition_test.py 名字 秒数 "ini 行;ini 行" [--size 1920 1080] [--mode 2|1] [--sample 40] [--noproxy]

  例:  superposition_test.py off 40 "Enabled=0" --size 1920 1080                 关掉 DLSS5 的帧率
       superposition_test.py s35 40 "WorkingScale=0.35" --size 1920 1080         开着的帧率与 GPU 负载
       superposition_test.py cap 40 "WorkingScale=1.0;DumpFrame=420"             存一帧处理前后的画面 (给 compare_frames.py)
       superposition_test.py soak 600 "WorkingScale=0.35" --mode 1 --sample 40   长时间稳定性 (内存、句柄、显存)
       superposition_test.py ctl 20 "Enabled=0" --noproxy                        对照: 暂时拿掉代理 dxgi.dll

准备: 把 Superposition 解到 _dl/unigine/Superposition (安装包是 Inno Setup，静默安装会以代码 2 退出，用 innoextract -e 解包即可)，
用管理器装上兜底模式 (dlss5 install <目录>)。输出: 帧率 (代理日志里每 10 秒一行的 presents 计数)、GPU 负载与功耗、
进程空闲的时间段、内存 / 句柄 / 显存随时间的变化；转储与日志移到 test/sp_<名字>/。

这个程序的几个脾气 (都实测过，与代理无关):
  - 窗口失去激活时它自己停止渲染: Present 不再被调用、所有线程 0% CPU、窗口并没有"无响应"。本脚本用一个线程不断向它的窗口投递
    WM_ACTIVATEAPP / WM_ACTIVATE / WM_SETFOCUS，让它以为自己是活动窗口 (不抢用户的焦点)。跑分模式下每秒 2 次就够；
    自由漫游模式下每次激活只渲染一两帧 (每秒 2 次时只有 4 fps)，所以默认每秒 100 次 (--keep-hz)。
  - 跑分模式 (-mode 2) 约 150 秒跑完后自己退出 (退出码 0)；自由漫游 (-mode 1) 不退出，长时间测试用它。
  - 它的输出很多，标准输出必须接到文件: 接到没人读的管道会把它卡住。
  - 运行期间用户按 ESC 会结束跑分。
"""
import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from ctypes import wintypes

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
user32 = ctypes.windll.user32
EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)


def windows_of(pid):
    """进程的可见顶层窗口"""
    found = []

    def cb(h, _):
        q = wintypes.DWORD()
        user32.GetWindowThreadProcessId(h, ctypes.byref(q))
        if q.value == pid and user32.IsWindowVisible(h) and user32.GetWindow(h, 4) == 0:
            found.append(h)
        return True
    user32.EnumWindows(EnumProc(cb), 0)
    return found


def foreground_pid():
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
    return pid.value


def smi(query):
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [float(x) for x in r.stdout.strip().split(",")]


def process_vram_mb(pid):
    """这个进程自己的专用显存 (Windows 性能计数器；nvidia-smi 在 WDDM 下只有全机总量)"""
    cmd = (r"(Get-Counter '\GPU Process Memory(pid_%d*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples"
           r" | Measure-Object CookedValue -Sum | ForEach-Object { [int]($_.Sum / 1MB) }") % pid
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()
    return int(out) if out else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("seconds", type=float, help="世界加载完之后再运行多久")
    ap.add_argument("ini", help="dlss5fb.ini 的 [DLSS5] 节内容，分号分隔")
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "..", "_dl", "unigine", "Superposition", "bin"))
    ap.add_argument("--size", nargs=2, default=["1280", "720"])
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--quality", default="1", help="shaders_quality (0 低 .. 3 极高)")
    ap.add_argument("--mode", default="2", help="2 = 跑分 (约 150 秒后自己退出)，1 = 自由漫游 (不退出)")
    ap.add_argument("--sample", type=float, default=0, help="每隔这么多秒记一次内存 / 句柄 / 显存")
    ap.add_argument("--keep-hz", type=float, default=100, help="每秒向窗口投递多少次激活消息 (0 = 不投递)")
    ap.add_argument("--noproxy", action="store_true", help="对照: 运行期间把代理 dxgi.dll 改名")
    a = ap.parse_args()
    BIN = os.path.abspath(a.bin)
    out = os.path.join(HERE, "test", "sp_" + a.name)
    os.makedirs(out, exist_ok=True)

    for f in os.listdir(BIN):
        if f.startswith("dlss5fb_") or f == "dlss5fb.log":
            os.remove(os.path.join(BIN, f))
    open(os.path.join(BIN, "dlss5fb.ini"), "w", encoding="ascii").write("[DLSS5]\n" + a.ini.replace(";", "\n") + "\n")
    args = [os.path.join(BIN, "superposition.exe"), "-sound_app", "null", "-system_script", "superposition/system_script.cpp",
            "-data_path", "../", "-engine_config", "../data/superposition/unigine.cfg", "-video_mode", "-1",
            "-project_name", "Superposition", "-video_resizable", "1",
            "-console_command", "config_readonly 1 && world_load superposition/superposition",
            "-mode", a.mode, "-preset", "0", "-video_width", a.size[0], "-video_height", a.size[1],
            "-video_fullscreen", "1" if a.fullscreen else "0", "-shaders_quality", a.quality, "-textures_quality", "1",
            "-dof", "1", "-motion_blur", "1", "-video_app", "direct3d11"]
    dx, dx_off = os.path.join(BIN, "dxgi.dll"), os.path.join(BIN, "dxgi.dll.off")
    if a.noproxy and os.path.exists(dx):
        os.replace(dx, dx_off)
    stdout_path = os.path.join(out, "stdout.txt")
    logf = open(stdout_path, "wb")
    p = subprocess.Popen(args, cwd=BIN, stdout=logf, stderr=subprocess.STDOUT)
    try:
        ps = psutil.Process(p.pid)
        t_launch = time.time()                       # 世界加载完才开始计时
        while time.time() - t_launch < 180 and p.poll() is None:
            time.sleep(0.5)
            text = open(stdout_path, "rb").read()
            if b"Benchmark running" in text or b"xinput1_4.dll" in text:
                break
        load_s = time.time() - t_launch
        ps.cpu_percent(None)
        stop = threading.Event()

        def keep_active():                           # 让它以为自己是活动窗口，否则它会暂停渲染
            while not stop.is_set() and p.poll() is None:
                for h in windows_of(p.pid):
                    for msg, wp in ((0x001C, 1), (0x0006, 1), (0x0007, 0)):
                        user32.PostMessageW(h, msg, wp, 0)
                stop.wait(1.0 / a.keep_hz)
        if a.keep_hz > 0:
            threading.Thread(target=keep_active, daemon=True).start()
        t0, start_clock = time.time(), time.localtime()
        cpu, fg, gpu, samples = [], [], [], []
        next_sample = a.sample
        while time.time() - t0 < a.seconds and p.poll() is None:
            time.sleep(0.5)
            fg.append(foreground_pid() == p.pid)
            try:
                cpu.append(ps.cpu_percent(None))
                if len(fg) % 4 == 0:
                    gpu.append(smi("utilization.gpu,power.draw"))
                if a.sample and time.time() - t0 >= next_sample:
                    next_sample += a.sample
                    m = ps.memory_info()
                    samples.append((time.time() - t0, m.rss / 2 ** 20, m.private / 2 ** 20, ps.num_handles(),
                                    smi("memory.used")[0], process_vram_mb(p.pid)))
            except (psutil.Error, OSError, ValueError):
                pass
        stop.set()
        code = p.poll()
        if code is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    finally:
        logf.close()
        if a.noproxy and os.path.exists(dx_off):
            os.replace(dx_off, dx)

    logp = os.path.join(BIN, "dlss5fb.log")
    log = open(logp, encoding="utf-8", errors="replace").read() if os.path.exists(logp) else ""
    pts = [(int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000, int(n))
           for h, mi, s, ms, n in re.findall(r"(\d\d):(\d\d):(\d\d)\.(\d+) presents: (\d+) total", log)]
    start = start_clock.tm_hour * 3600 + start_clock.tm_min * 60 + start_clock.tm_sec
    fps = [(b[1] - x[1]) / (b[0] - x[0]) for x, b in zip(pts, pts[1:]) if x[0] >= start]
    line = f"{a.name}: 加载 {load_s:.0f} s；" + ("仍在运行" if code is None else f"自己退出了 (代码 {code})")
    if fps:
        line += f"；帧率 平均 {sum(fps) / len(fps):.1f} (最低 {min(fps):.0f}，最高 {max(fps):.0f}，{len(fps)} 个 10 秒窗口)"
    print(line)
    if gpu:
        print(f"    GPU 负载 平均 {sum(g[0] for g in gpu) / len(gpu):.0f}% (最低 {min(g[0] for g in gpu):.0f})，功耗 平均 {sum(g[1] for g in gpu) / len(gpu):.0f} W；"
              f"进程 CPU 平均 {sum(cpu) / max(len(cpu), 1):.0f}%；窗口在前台的时间 {100 * sum(fg) / max(len(fg), 1):.0f}%")
    idle, run = [], []                               # 进程空闲 (CPU < 5%) 3 秒以上的时间段
    for k, c in enumerate(cpu + [100]):
        if c < 5:
            run.append(k)
        else:
            if len(run) >= 6:
                idle.append((run[0] * 0.5, run[-1] * 0.5))
            run = []
    for x, y in idle:
        print(f"    空闲 {x:.0f}-{y:.0f} s (没有渲染)")
    if any(f < 5 for f in fps):
        print("    有 10 秒窗口的帧率低于 5: 渲染停过")
    for key in ("tracked", "FAIL", "resize:", "not supported", "device removed", "picture:"):
        for l in [l for l in log.splitlines() if key in l][:2]:
            print("   ", l[13:150])
    print(f"    场景切换 {len([l for l in log.splitlines() if 'scene cut' in l])} 次")
    if samples:
        print("    时间 s | 工作集 MB | 私有内存 MB | 句柄 | 全机显存 MB | 本进程显存 MB")
        step = max(1, len(samples) // 14)
        for s in samples[::step] + ([samples[-1]] if (len(samples) - 1) % step else []):
            print("    %6.0f | %8.0f | %8.0f | %5d | %7.0f | %6d" % s)
        first, last = samples[min(1, len(samples) - 1)], samples[-1]
        print(f"    从第 {first[0]:.0f} 秒到结束: 私有内存 {last[2] - first[2]:+.0f} MB，句柄 {last[3] - first[3]:+d}，本进程显存 {last[5] - first[5]:+d} MB")
    for f in os.listdir(BIN):
        if f.startswith("dlss5fb_") and not f.endswith(".bin"):
            shutil.move(os.path.join(BIN, f), os.path.join(out, f))
    if os.path.exists(logp):
        shutil.copy(logp, os.path.join(out, "dlss5fb.log"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
