# 从原件重建 310.9.16 十月实验版宿主

用原件可复算 286,873,600 字节的实验 DLL，SHA-256 为 `2d3e405535cfb574828a5940f62d14952044f4feb1013caf9e7b9240385506ed`。构建工具只读输入，将生成物放在 `_dl/`。15 个模块的预期哈希和地址见[静态证据](validation/october_nr_staged_host.json)，RTX 3060 结果见[整网实测](october_nr_ampere_runtime.md)。

准备 RTX Remix 1.6.1 的官方 Windows x64 ZIP，把归档根部的 `nvngx_dlssnr.dll` 提取到本地忽略目录。ZIP SHA-256 为 `7dd1d44492a609050e44b0e3581cec31135d1ed88b4a91123b4faae91955716a`，DLL SHA-256 为 `2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f`。[官方发布](https://github.com/NVIDIAGameWorks/rtx-remix/releases/tag/remix-1.6.1) · [签名与校验清单](validation/october_nr_remix_provenance.json)

准备已有 310.8.0 双架构 fatbin `fatbin_08.sm86.fatbin` 至 `fatbin_15.sm86.fatbin`。这些模块对应的两版原始 fatbin 逐字节相同。它们通常在 `sm86_port/`，也可使用以前完整重建保留的等字节文件。

以下从仓库根目录运行，需要 Python 3.11+ 与 CUDA **13.3.73** 的三个 Windows 工具，默认工具放在 `cuda_tools/bin/`。将两个输出路径设为仓库 `_dl/` 下尚未创建的直接子目录。

```powershell
$python = 'python'
$original = '_dl/october_nr_20261106/original/nvngx_dlssnr.dll'
$baseline = 'sm86_port'
$inputs = '_dl/october-replay-inputs'
$output = '_dl/october-replay-stage'
```

## 1. 准备并核对 15 项双架构输入

前七项按原 DLL 的模块地址提取 fatbin，用 `cuobjdump` 分离 PTX 和原 sm120 ELF，再用默认改写配置和 `ptxas -arch=sm_86` 编译。原实验的模块 7 在模块 5 之后复用同一改写进程，内部拷贝标签从 105 编号。并行重放时为模块 7 恢复计数 104，其他模块从零开始。模块 6 的临时 PTX 副本将恰好 21 处 `fence.acquire.gpu` 改为 `fence.acq_rel.gpu`。前七项按 **sm86、sm120** 顺序打包，后八项复用基线包。每项编译或复用结果都核对哈希。

```powershell
@'
from concurrent.futures import ThreadPoolExecutor
import json, mmap, os, subprocess, sys
from pathlib import Path
from tools.inspect_nr_port_host import OCTOBER_SHA, inspect_pe, require, sha
from tools.ptx86.stage_october_host import KERNEL_REPORT, KERNEL_REPORT_SHA, dual_elf_hashes, output_dir

root = Path.cwd()
original = Path(sys.argv[1]).resolve(strict=True)
baseline = Path(sys.argv[2]).resolve(strict=True)
work = output_dir(Path(sys.argv[3]))
pins = {
    'cuda_tools/bin/cuobjdump.exe': 'cb72e7e353ed20320deb4770f75291a828c0b23ca8c2d27688c4eeced96543f5',
    'cuda_tools/bin/ptxas.exe': '86b2ff48655c55f1547edc59b503d39a62aa5c71745b7ecf6be526c29af0b693',
    'cuda_tools/bin/fatbinary.exe': '7ac13247edd275418ac8d8f4dd41ae1900a52b58b4f289a55956dda3ac613715',
    'tools/ptx86/rewrite.py': 'acca39bfcbb6b5644e26defe82815e17d11e2a2f0f7e92df08df33bb1d4274fa',
    'tools/ptx86/fuse.py': '4c4fc4b32970e9b269c874630f1b3c9f204222677f9db4fe814745dbbcdd87a8',
    'tools/ptx86/flow.py': '1b66a27564d2869d06fe6b22386cf5083380a854d60f291c234fc7d4e025a1b8',
    'tools/inspect_nr_port_host.py': 'f40f0ac5d81545600e31ba444099f0ce8ac25a0d675aaa4a14876f4a047a13f3',
    'tools/ptx86/stage_october_host.py': '9b0471b771e9d97a3ee89efe38751446b6bc1592b8a211822f84f5fc2e53351e',
    'research/validation/october_nr_staged_host.json': '72d88471d4f87743a5ee719ec4494ef2f7d76a78d033eb4e586c0a7e162c868b',
}
for path, expected in pins.items():
    require(sha((root / path).read_bytes()) == expected, f'Input changed: {path}')
proof = json.loads((root / 'research/validation/october_nr_staged_host.json').read_bytes())
kernel_bytes = KERNEL_REPORT.read_bytes()
require(sha(kernel_bytes) == KERNEL_REPORT_SHA, 'Kernel evidence changed')
kernel = json.loads(kernel_bytes)
require(kernel['candidate_dll_sha256'] == OCTOBER_SHA, 'Kernel evidence source mismatch')
rewritten = {
    1: '0745e256618fd9a8224dbfd48a82c3cc91010c6ef36a43af4225849cbe57f352',
    2: 'bddbaaeecac0ef58ef87f3bc631b245118708e58a21bcb1e7e399e5a223c2676',
    3: '6f6e1118b28377afeb6345710017fd4e13986e2a7d5b4e82f79b1d3f0fecad96',
    4: '2ff3412ba6fb03525c9f0e6b5fb4d40231fc4b25ed39a4fcb8d17c3f22ef6d5f',
    5: 'dea5cee7b1aef0e0cc534814f6f0126340de600f23a9a053957972d58b916e25',
    6: 'bf184bd34608a67997d2fd426de5b59f88e9bf84e96e928ff9d9e7fba9b71aae',
    7: '8800bc98182a7f0a87d7a977bbeb8b82fd6633207de68627f92c7060045e14c4',
}

def run(*args, cwd):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)

tools = root / 'cuda_tools/bin'
with original.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
    host = inspect_pe(raw, OCTOBER_SHA, october=True)
    work.mkdir(exist_ok=False)
    for row in host['modules'][:7]:
        n = row['module']
        folder = work / f'module-{n:02d}'
        folder.mkdir()
        off = int(row['file_offset'], 16)
        source_fatbin = raw[off:off + row['bytes']]
        require(sha(source_fatbin) == row['sha256'], f'Original module {n} changed')
        with (folder / f'fatbin_{n:02d}.fatbin').open('xb') as destination:
            destination.write(source_fatbin)

def prepare(n):
    expected = proof['bindings'][n - 1]
    if n <= 7:
        folder = work / f'module-{n:02d}'
        tag = f'fatbin_{n:02d}'
        source_fatbin = folder / f'{tag}.fatbin'
        run(tools / 'cuobjdump.exe', '--extract-ptx', 'all', source_fatbin, cwd=folder)
        run(tools / 'cuobjdump.exe', '--extract-elf', 'all', source_fatbin, cwd=folder)
        ptx, native = (folder / f'{tag}.1.sm_120.{ext}' for ext in ('ptx', 'cubin'))
        info = kernel['modules'][n - 1]['candidate']
        require(sha(ptx.read_bytes()) == info['ptx_sha256'] and sha(native.read_bytes()) == expected['sm120_sha256'],
                f'Original PTX/cubin extraction changed for module {n}')
        rewritten_ptx = folder / f'module{n:02d}.sm86.ptx'
        if n == 7:
            # Modules 5 and 7 originally shared the label counter. Restore it
            # only in this subprocess to reproduce the original PTX bytes.
            driver = ('from tools.ptx86 import rewrite as r; '
                      'r._label[0] = 104; r.main()')
            run(sys.executable, '-B', '-c', driver, ptx, rewritten_ptx, cwd=root)
        else:
            run(sys.executable, '-B', '-m', 'tools.ptx86.rewrite', ptx, rewritten_ptx, cwd=root)
        data = rewritten_ptx.read_bytes()
        require(sha(data) == rewritten[n], f'Rewritten PTX differs for module {n}')
        if n == 6:
            require(data.count(b'fence.acquire.gpu') == 21, 'Unexpected fence count')
            data = data.replace(b'fence.acquire.gpu', b'fence.acq_rel.gpu')
            require(sha(data) == 'f6bd97ebc3db9d9c190e731144005ec2bda23b147325bcdbf77f784007a88f29',
                    'Module 6 experiment differs')
            rewritten_ptx = folder / 'module06.experimental-acq-rel.ptx'
            with rewritten_ptx.open('xb') as trial:
                trial.write(data)
        compiled = rewritten_ptx.with_suffix('.cubin')
        run(tools / 'ptxas.exe', '-arch=sm_86', rewritten_ptx, '-o', compiled, cwd=folder)
        require(sha(compiled.read_bytes()) == expected['sm86_sha256'], f'sm86 output differs for module {n}')
        fatbin = folder / f'module{n:02d}.fatbin'
        run(tools / 'fatbinary.exe', '-64', '--compress=false', f'--create={fatbin}',
            f'--image3=kind=elf,sm=86,file={compiled}', f'--image3=kind=elf,sm=120,file={native}', cwd=folder)
    else:
        fatbin = baseline / f'fatbin_{n:02d}.sm86.fatbin'
    data = fatbin.read_bytes()
    digest = sha(data)
    require(digest == expected['sha256'] and dual_elf_hashes(data) == {86: expected['sm86_sha256'], 120: expected['sm120_sha256']},
            f'Dual-architecture fatbin differs for module {n}')
    print(f'Module {n:02d} matches', flush=True)
    return dict(module=n, fatbin=os.path.relpath(fatbin, work), fatbin_sha256=digest,
                sm86_sha256=expected['sm86_sha256'])

with ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(prepare, range(1, 16)))
with (work / 'modules.json').open('x', encoding='utf-8', newline='\n') as manifest:
    manifest.write(json.dumps(dict(schema=1, source_sha256=OCTOBER_SHA, modules=rows), indent=2) + '\n')
'@ | & $python -B - $original $baseline $inputs
if ($LASTEXITCODE -ne 0) { throw 'Module preparation failed; preserve its output and stop.' }
```

## 2. 暂存、重读并独立复验

默认按原版架构门槛构建。下列实验候选用 `--allow-ampere-experiment` 改写 0x170 的一个选择字节。构建和磁盘复验均解析 13 个 RIP 相对引用及长度、2 个 DIR64 绝对引用，并核对 PE 校验和、完整的 `.rsrc` 与 `.reloc`、门槛代码、其他选择字节和两个架构的模块哈希。

```powershell
$modules = Join-Path $inputs 'modules.json'
& $python -B -m tools.ptx86.stage_october_host --original $original --modules $modules --out $output --allow-ampere-experiment
if ($LASTEXITCODE -ne 0) { throw 'Stage failed.' }
$staged = Join-Path $output 'nvngx_dlssnr.staged.dll'
& $python -B -m tools.ptx86.stage_october_host --original $original --modules $modules --verify-only $staged --allow-ampere-experiment
if ($LASTEXITCODE -ne 0) { throw 'Independent static verification failed.' }
$expected = '2d3e405535cfb574828a5940f62d14952044f4feb1013caf9e7b9240385506ed'
if ((Get-FileHash -LiteralPath $staged -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Staged candidate differs from the recorded experiment.'
}
```

以上命令复现固定输入和工具下的候选字节。跨架构的同步和数值等价由独立动态测试验收。
