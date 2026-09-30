"""71 个 block 的镜像拓扑。

这部分是**实测**的：由参数量的完美镜像 + 边界块偏大 + kernel 变体名共同确定。
但「激活的层序列」本身没有序列化在 DLL 里（builder 走描述符 + 工厂分派），
所以下面给出的是**结构**，不是被证实的执行顺序。
"""

#: (block 范围, 宽度, 家族, 角色)
TOPOLOGY = [
    ((0, 0),   32,  "single1h", "输入适配器 16→32 + 1H block"),
    ((1, 3),   32,  "single1h", "1H 编码器"),
    ((4, 4),   32,  "single1h", "1H → 2H 下采样过渡"),
    ((5, 7),   64,  "single",   "2H 编码器"),
    ((8, 8),   64,  "single",   "2H → 4H 下采样过渡"),
    ((9, 13),  128, "single",   "4H 编码器"),
    ((14, 14), 128, "single",   "4H → 8H 下采样过渡"),
    ((15, 21), 256, "single",   "8H 编码器"),
    ((22, 22), 256, "single",   "8H → 16H 下采样过渡"),
    ((23, 29), 256, "cuckoo4",  "CrazyCuckoo 编码器（split ×2）"),
    ((30, 30), 256, "cuckoo5",  "核前边界组（5 记录）"),
    ((31, 38), 512, "split16h", "16H 中央核 ×8（占 68% 参数）"),
    ((39, 39), 512, "bridge",   "decoder 输入过渡 512×512"),
    ((40, 47), 256, "cuckoo4",  "CrazyCuckoo 解码器（split ×2）"),
    ((48, 48), 256, "single",   "16H → 8H 上采样过渡"),
    ((49, 55), 256, "single",   "8H 解码器"),
    ((56, 56), 128, "single",   "8H → 4H 上采样过渡"),
    ((57, 61), 128, "single",   "4H 解码器"),
    ((62, 62), 64,  "single",   "4H → 2H 上采样过渡"),
    ((63, 65), 64,  "single",   "2H 解码器"),
    ((66, 66), 32,  "single1h", "2H → 1H 上采样过渡"),
    ((67, 69), 32,  "single1h", "1H 解码器"),
    ((70, 70), 32,  "output",   "post-block + blend_scale"),
]

#: 编码器 ↔ 解码器的镜像配对（用于 skip）
SKIP_PAIRS = [((0, 4), (66, 70)), ((5, 8), (62, 65)), ((9, 14), (56, 61)),
              ((15, 22), (48, 55)), ((23, 29), (40, 47))]

#: 已确认的接口宽度
INPUT_CHANNELS = 16       # block0 多出的 512 = 16×32
HEAD_DIM = 32             # 四组模板配置互证
WINDOW_TOKENS = 64        # 8×8 tile
CORE_EXPAND = (512, 1024)  # FinalHead
BRIDGE = (1024, 512)      # CCDecInputUpsample


def describe():
    print("%-12s %5s %-10s %s" % ("block", "W", "家族", "角色"))
    print("-" * 68)
    for (lo, hi), W, fam, role in TOPOLOGY:
        rng = "block%d" % lo if lo == hi else "block%d-%d" % (lo, hi)
        print("%-12s %5d %-10s %s" % (rng, W, fam, role))
    print("\n跳连配对（编码器 → 解码器）:")
    for a, b in SKIP_PAIRS:
        print("  block%d-%d  →  block%d-%d" % (a[0], a[1], b[0], b[1]))
    print("\n解码器 skip 合并方式 = element-wise add  [实测 fatbin_07.ptx]")
    print("重叠 tile 由 red.global.v4.f16x2.add.noftz 原子加归约")


if __name__ == "__main__":
    describe()
