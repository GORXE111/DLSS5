"""dllq — DLL / PTX 结构化检索。给 agent 用的。

    python -m dllq <命令> [参数]

先跑一次 `python -m dllq index` 建索引，之后所有查询都是秒级。
"""
import argparse, sys
from .core import out_utf8


HELP = """dllq — 按「你想问什么」组织的命令表

  想知道整体有什么          dllq map
  这个数在哪出现            dllq const 0.894531
  这个 kernel 长什么样      dllq kernel vit_attention -d
  它内部用了哪些常数        dllq consts-in vit_attention
  谁用了这条指令            dllq ops rsqrt.approx
  这个寄存器在哪定值        dllq def %r2677 --kernel pre_block
  在 PTX 里搜正则           dllq grep "shl.b32 .*, 4" --fatbin 6
  二进制里有这个串吗        dllq str SkinStructure
  权重是怎么分布的          dllq weights            /  dllq weights --block 31
  换个架构能编过吗          dllq isa sm_86
  一条记录里打包了什么      dllq unpack block23.layer2.layer --width 256
  这些层是什么形状          dllq shape          /  dllq shape --weights --block 31
  记一条发现                dllq note add "..." --cite fatbin_06.ptx:20950
  之前查到过什么            dllq note ls            /  dllq note find softmax
  概念性问题（语义）        dllq vec search "哪段在做归一化"
  不确定该用哪个            dllq ask "0x3FFC4000 是什么"

所有输出都带 fatbin_NN.ptx:行号 引用，可直接 sed -n 取原文。
"""


def main(argv=None):
    out_utf8()
    from . import query as Q

    ap = argparse.ArgumentParser(prog="dllq", add_help=False,
                                 description="DLL / PTX 结构化检索")
    ap.add_argument("-h", "--help", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("index", help="建立/重建索引")
    sub.add_parser("map", help="全局地图")

    p = sub.add_parser("const", help="数值反查")
    p.add_argument("value")

    p = sub.add_parser("kernel", help="kernel 概览")
    p.add_argument("name"); p.add_argument("-d", "--detail", action="store_true")

    p = sub.add_parser("consts-in", help="列出某 kernel 内的立即数")
    p.add_argument("name")

    p = sub.add_parser("ops", help="指令普查")
    p.add_argument("op")

    p = sub.add_parser("def", help="寄存器定值/使用")
    p.add_argument("reg"); p.add_argument("--kernel"); p.add_argument("--fatbin")

    p = sub.add_parser("grep", help="PTX 词法检索")
    p.add_argument("pattern"); p.add_argument("--fatbin")
    p.add_argument("-A", type=int, default=0); p.add_argument("-B", type=int, default=0)
    p.add_argument("-n", type=int, default=30); p.add_argument("-F", action="store_true")

    p = sub.add_parser("str", help="二进制字符串检索")
    p.add_argument("pattern"); p.add_argument("--cls"); p.add_argument("-n", type=int, default=30)

    p = sub.add_parser("weights", help="权重记录")
    p.add_argument("--block", type=int)

    p = sub.add_parser("isa", help="目标架构可编译性")
    p.add_argument("target")

    p = sub.add_parser("shape", help="层形状：模板配置 + 参数量反推")
    p.add_argument("--weights", action="store_true"); p.add_argument("--block", type=int)

    p = sub.add_parser("unpack", help="把权重记录切成子张量")
    p.add_argument("name"); p.add_argument("--width", type=int)
    p.add_argument("--thr", type=float, default=0.5)

    p = sub.add_parser("addr", help="tid=0 下求解权重 blob 内的常量偏移")
    p.add_argument("kernel"); p.add_argument("--param", type=int, default=16)

    p = sub.add_parser("dis", help="x86-64 反汇编（host 侧）")
    p.add_argument("addr"); p.add_argument("-n", type=int, default=60)
    p.add_argument("--strings", action="store_true", default=True)

    p = sub.add_parser("xref", help="找引用某字符串的代码位置")
    p.add_argument("text")

    p = sub.add_parser("note", help="发现日志")
    p.add_argument("action", choices=["add", "ls", "find"])
    p.add_argument("text", nargs="?")
    p.add_argument("--cite", nargs="*"); p.add_argument("--tag", nargs="*")

    p = sub.add_parser("vec", help="语义层（可选）")
    p.add_argument("action", choices=["build", "search"])
    p.add_argument("text", nargs="?"); p.add_argument("-k", type=int, default=8)
    p.add_argument("--src", choices=["note", "kernel", "str"])

    p = sub.add_parser("ask", help="不确定用哪个命令时的前门")
    p.add_argument("words", nargs="+")

    a = ap.parse_args(argv)
    if a.help or not a.cmd:
        print(HELP)
        return 0

    if a.cmd == "index":
        from .index import build
        build()
        return 0

    fn = {"map": Q.cmd_map, "const": Q.cmd_const, "kernel": Q.cmd_kernel,
          "consts-in": Q.cmd_consts_in, "ops": Q.cmd_ops, "def": Q.cmd_def,
          "grep": Q.cmd_grep, "str": Q.cmd_str, "weights": Q.cmd_weights,
          "isa": Q.cmd_isa, "shape": Q.cmd_shape, "unpack": Q.cmd_unpack, "addr": __import__("dllq.addr",fromlist=["x"]).cmd_addr,
          "dis": __import__("dllq.dis",fromlist=["x"]).cmd_dis,
          "xref": __import__("dllq.dis",fromlist=["x"]).cmd_xref, "vec": __import__("dllq.vec", fromlist=["x"]).cmd_vec, "note": Q.cmd_note, "ask": Q.cmd_ask}[a.cmd]
    fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
