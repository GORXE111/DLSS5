// sm_86 移植用的逐 kernel 计时器 (DLSS5_PROF=1 时启用)。
//
// nvngx_dlssnr.dll 经 GetProcAddress(nvapi64, "nvapi_QueryInterface") 取 NvAPI，
// 用 NvAPI_D3D12_CreateCuModule / CreateCuFunction 建 kernel，NvAPI_D3D12_LaunchCuKernelChain 成链发射。
// 这里钩它的 GetProcAddress，把 QueryInterface 换成包装:
//   CreateCuFunction    -> 记下 handle -> kernel 名
//   LaunchCuKernelChain -> 拆成逐个发射，每个前后各插一个 D3D12 时间戳查询
// 每帧提交前 Resolve，等完 fence 后累计，进程退出时按总耗时排序打印。
// 旧的 CreateCubinComputeShader* / LaunchCubinShader 接口也一并包装，dll 换版本时仍可用。
#pragma once
#include <d3d12.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace prof {

using Handle = void *;
using PfnQueryInterface = void *(__cdecl *)(unsigned);

struct Dim3 { unsigned x, y, z; };
struct CuLaunch {  // NVAPI_CU_KERNEL_LAUNCH_PARAMS
    Handle function;
    Dim3 grid, block;
    unsigned dyn_smem;
    const void *params;
    unsigned param_size;
};
using PfnCreateCuFunction = int(__cdecl *)(ID3D12Device *, Handle, const char *, Handle *);
using PfnLaunchCuKernelChain = int(__cdecl *)(ID3D12GraphicsCommandList *, const CuLaunch *, unsigned);
using PfnCreateCubinWithName = int(__cdecl *)(ID3D12Device *, const void *, unsigned, unsigned, unsigned, unsigned,
    const char *, Handle *);
using PfnLaunchCubin = int(__cdecl *)(ID3D12GraphicsCommandList *, Handle, unsigned, unsigned, unsigned, const void *,
    unsigned);

constexpr unsigned kIdCreateCuFunction = 0xe2436e22, kIdLaunchCuKernelChain = 0x24973538,
                   kIdCreateWithName = 0x1dc7261f, kIdLaunch = 0x5c52bb86;
constexpr UINT kMaxQueries = 16384;

struct Kernel { std::string name; unsigned bx = 0, by = 0, bz = 0, smem = 0; };
struct Launch { Handle h; unsigned gx, gy, gz; UINT q; };
struct Acc { double ms = 0; unsigned launches = 0; std::string name; unsigned gx, gy, gz, bx, by, bz, smem; };

inline bool g_on = false;
inline PfnQueryInterface g_real_qi;
inline PfnCreateCuFunction g_real_create_cu_function;
inline PfnLaunchCuKernelChain g_real_launch_chain;
inline PfnCreateCubinWithName g_real_create_named;
inline PfnLaunchCubin g_real_launch;
inline std::map<Handle, Kernel> g_kernels;
inline std::vector<Launch> g_launches;
inline std::map<std::string, Acc> g_acc;
inline std::vector<unsigned> g_ids;
inline ID3D12QueryHeap *g_heap;
inline ID3D12Resource *g_readback;
inline UINT g_next;
inline unsigned g_frames;
inline double g_span_ms;
inline std::vector<double> g_spans;  // 每帧 GPU 跨度，报告 min/中位数 (机器上有别的程序占 GPU 时平均值不可信)

inline bool EnsureHeap(ID3D12GraphicsCommandList *list)
{
    if (g_heap) return true;
    ID3D12Device *device = nullptr;
    if (FAILED(list->GetDevice(IID_PPV_ARGS(&device)))) return false;
    D3D12_QUERY_HEAP_DESC hd = {};
    hd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    hd.Count = kMaxQueries;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = kMaxQueries * sizeof(UINT64);
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const bool ok = SUCCEEDED(device->CreateQueryHeap(&hd, IID_PPV_ARGS(&g_heap))) &&
        SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&g_readback)));
    // 实测: 自建 D3D12 缓冲的 GPU VA (0x1e4xxxxx) 与 dll 参数里的指针同一区间 -> 那些指针就是 D3D12 GPU VA
    device->Release();
    return ok;
}

inline int __cdecl CreateCuFunction(ID3D12Device *d, Handle module, const char *name, Handle *h)
{
    const int r = g_real_create_cu_function(d, module, name, h);
    if (r == 0 && h) g_kernels[*h].name = name ? name : "?";
    return r;
}

// 执行轨迹 (DLSS5_TRACE=帧号): 该帧每次发射记一行 —— 序号 链号/链内位置 kernel grid block smem 参数字节(hex)
inline unsigned g_frame_idx;
inline int g_trace_frame = -1;
inline FILE *g_trace;
inline unsigned g_trace_seq, g_trace_chain;

inline void TraceChain(const CuLaunch *k, unsigned n)
{
    if (g_trace_frame < 0 || g_frame_idx != unsigned(g_trace_frame)) return;
    if (!g_trace) fopen_s(&g_trace, "nr-trace.tsv", "w");
    if (!g_trace) return;
    for (unsigned i = 0; i < n; ++i) {
        const Kernel &kn = g_kernels[k[i].function];
        std::fprintf(g_trace, "%u\t%u/%u/%u\t%s\t%u,%u,%u\t%u,%u,%u\t%u\t%u\t", g_trace_seq++, g_trace_chain, i, n,
            kn.name.c_str(), k[i].grid.x, k[i].grid.y, k[i].grid.z, k[i].block.x, k[i].block.y, k[i].block.z,
            k[i].dyn_smem, k[i].param_size);
        const auto *p = static_cast<const unsigned char *>(k[i].params);
        for (unsigned b = 0; p && b < k[i].param_size; ++b) std::fprintf(g_trace, "%02x", p[b]);
        std::fprintf(g_trace, "\n");
    }
    g_trace_chain++;
    std::fflush(g_trace);
}

// ---------------------------------------------------------------- 显存窃听
// DLSS5_TAP="帧号;序号:地址:字节数;..."  地址可写 p<i> = 该次发射参数块里第 i 个 64 位字 (指针)。
// 在该帧第 <序号> 次发射之后插一个 tap_copy kernel (tap.cubin，与 nr-lab.exe 同目录)，把显存拷进自有缓冲，
// 帧结束后写成 tap_s<序号>_<地址>.bin，并在 tap_index.tsv 记一行。
using PfnCreateCuModule = int(__cdecl *)(ID3D12Device *, const void *, unsigned, Handle *);
constexpr unsigned kIdCreateCuModule = 0xad1a677d;
struct TapReq { unsigned seq; std::string addr; size_t bytes; };
struct TapDone { unsigned seq; unsigned long long addr; size_t bytes, offset; };
inline std::vector<TapReq> g_tap_reqs;
inline int g_tap_frame = -1;
inline std::vector<TapDone> g_tap_done;
inline Handle g_tap_fn;
inline ID3D12Resource *g_tap_buf, *g_tap_rb;
inline size_t g_tap_cap, g_tap_used;
inline unsigned g_launch_seq;

inline void ParseTap()
{
    const char *spec = std::getenv("DLSS5_TAP");
    if (!spec) return;
    g_tap_frame = std::atoi(spec);
    for (const char *p = std::strchr(spec, ';'); p; p = std::strchr(p + 1, ';')) {
        TapReq t;
        t.seq = unsigned(std::strtoul(p + 1, nullptr, 0));
        const char *a = std::strchr(p + 1, ':');
        const char *b = a ? std::strchr(a + 1, ':') : nullptr;
        if (!b) break;
        t.addr.assign(a + 1, b);
        t.bytes = std::strtoull(b + 1, nullptr, 0);
        g_tap_reqs.push_back(t);
        g_tap_cap += (t.bytes + 255) / 256 * 256;
    }
}

inline bool EnsureTap(ID3D12GraphicsCommandList *list)
{
    if (g_tap_fn) return true;
    ID3D12Device *device = nullptr;
    if (FAILED(list->GetDevice(IID_PPV_ARGS(&device)))) return false;
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::strcpy(std::strrchr(path, '\\') + 1, "tap.cubin");
    FILE *f = nullptr;
    fopen_s(&f, path, "rb");
    std::vector<char> blob;
    if (f) {
        fseek(f, 0, SEEK_END);
        blob.resize(size_t(ftell(f)));
        fseek(f, 0, SEEK_SET);
        fread(blob.data(), 1, blob.size(), f);
        fclose(f);
    }
    auto create_module = reinterpret_cast<PfnCreateCuModule>(g_real_qi(kIdCreateCuModule));
    Handle mod = nullptr;
    const int rm = create_module && !blob.empty() ? create_module(device, blob.data(), unsigned(blob.size()), &mod) : -1;
    const int rf = rm == 0 ? g_real_create_cu_function(device, mod, "tap_copy", &g_tap_fn) : -1;
    D3D12_HEAP_PROPERTIES hp = {};
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = g_tap_cap;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    const bool b1 = SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_tap_buf)));
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    const bool b2 = SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_tap_rb)));
    device->Release();
    std::printf("[tap] cubin %zu B, CreateCuModule=%d CreateCuFunction=%d, buffer %zu B %s\n", blob.size(), rm, rf,
        g_tap_cap, b1 && b2 ? "ok" : "FAILED");
    if (rf != 0 || !b1 || !b2) g_tap_fn = nullptr;
    return g_tap_fn != nullptr;
}

inline void UavBarrier(ID3D12GraphicsCommandList *list)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &b);
}

// 第 seq 次发射刚完成: 按需插 tap
inline void TapAfter(ID3D12GraphicsCommandList *list, unsigned seq, const CuLaunch &launched)
{
    if (g_tap_frame < 0 || g_frame_idx != unsigned(g_tap_frame)) return;
    for (const TapReq &t : g_tap_reqs) {
        if (t.seq != seq || !EnsureTap(list)) continue;
        unsigned long long addr = 0;
        if (t.addr[0] == 'p') {
            const unsigned idx = unsigned(std::atoi(t.addr.c_str() + 1));
            if ((idx + 1) * 8 > launched.param_size) continue;
            std::memcpy(&addr, static_cast<const char *>(launched.params) + idx * 8, 8);
        } else {
            addr = std::strtoull(t.addr.c_str(), nullptr, 0);
        }
        struct { unsigned long long src, dst; unsigned n16, pad; } prm = {
            addr, g_tap_buf->GetGPUVirtualAddress() + g_tap_used, unsigned((t.bytes + 15) / 16), 0 };
        CuLaunch l = {};
        l.function = g_tap_fn;
        l.grid = { std::min(1024u, (prm.n16 + 255) / 256), 1, 1 };
        l.block = { 256, 1, 1 };
        l.params = &prm;
        l.param_size = 20;
        UavBarrier(list);
        const int r = g_real_launch_chain(list, &l, 1);
        UavBarrier(list);
        std::printf("[tap] frame %u seq %u 0x%llx +%zu -> off %zu (launch=%d)\n", g_frame_idx, seq, addr, t.bytes,
            g_tap_used, r);
        g_tap_done.push_back(TapDone{ seq, addr, t.bytes, g_tap_used });
        g_tap_used += (t.bytes + 255) / 256 * 256;
    }
}

inline void TapResolve(ID3D12GraphicsCommandList *list)
{
    if (g_tap_used && !g_tap_done.empty()) list->CopyBufferRegion(g_tap_rb, 0, g_tap_buf, 0, g_tap_used);
}

inline void TapCollect()
{
    if (!g_tap_used || g_tap_done.empty()) return;
    unsigned char *p = nullptr;
    D3D12_RANGE range{ 0, g_tap_used };
    if (SUCCEEDED(g_tap_rb->Map(0, &range, reinterpret_cast<void **>(&p)))) {
        FILE *idx = nullptr;
        fopen_s(&idx, "tap_index.tsv", "a");
        for (const TapDone &d : g_tap_done) {
            char path[96];
            sprintf_s(path, "tap_s%u_%llx.bin", d.seq, d.addr);
            FILE *f = nullptr;
            fopen_s(&f, path, "wb");
            if (f) { fwrite(p + d.offset, 1, d.bytes, f); fclose(f); }
            if (idx) std::fprintf(idx, "%u\t0x%llx\t%zu\t%s\n", d.seq, d.addr, d.bytes, path);
        }
        if (idx) fclose(idx);
        D3D12_RANGE none{ 0, 0 };
        g_tap_rb->Unmap(0, &none);
    }
    g_tap_done.clear();
    g_tap_used = 0;
}

// 链拆成逐个发射，每个前后各一个时间戳 (同一命令列表内顺序不变)
inline int __cdecl LaunchCuKernelChain(ID3D12GraphicsCommandList *list, const CuLaunch *k, unsigned n)
{
    // 初始化时 dll 会以空命令列表调一次 (能力探测)，原样透传
    if (!g_on || list == nullptr || k == nullptr || !EnsureHeap(list)) return g_real_launch_chain(list, k, n);
    TraceChain(k, n);
    int r = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (g_next + 2 > kMaxQueries) return g_real_launch_chain(list, k + i, n - i);
        const UINT q = g_next;
        g_next += 2;
        list->EndQuery(g_heap, D3D12_QUERY_TYPE_TIMESTAMP, q);
        r = g_real_launch_chain(list, k + i, 1);
        list->EndQuery(g_heap, D3D12_QUERY_TYPE_TIMESTAMP, q + 1);
        Kernel &kn = g_kernels[k[i].function];
        kn.bx = k[i].block.x;
        kn.by = k[i].block.y;
        kn.bz = k[i].block.z;
        kn.smem = k[i].dyn_smem;
        g_launches.push_back(Launch{ k[i].function, k[i].grid.x, k[i].grid.y, k[i].grid.z, q });
        TapAfter(list, g_launch_seq++, k[i]);
        if (r != 0) break;
    }
    return r;
}

inline int __cdecl CreateWithName(ID3D12Device *d, const void *c, unsigned s, unsigned x, unsigned y, unsigned z,
    const char *n, Handle *h)
{
    const int r = g_real_create_named(d, c, s, x, y, z, n, h);
    if (r == 0 && h) g_kernels[*h] = Kernel{ n ? n : "?", x, y, z, 0 };
    return r;
}

inline int __cdecl LaunchCubin(ID3D12GraphicsCommandList *list, Handle h, unsigned gx, unsigned gy, unsigned gz,
    const void *params, unsigned param_size)
{
    if (!g_on || list == nullptr || !EnsureHeap(list) || g_next + 2 > kMaxQueries)
        return g_real_launch(list, h, gx, gy, gz, params, param_size);
    const UINT q = g_next;
    g_next += 2;
    list->EndQuery(g_heap, D3D12_QUERY_TYPE_TIMESTAMP, q);
    const int r = g_real_launch(list, h, gx, gy, gz, params, param_size);
    list->EndQuery(g_heap, D3D12_QUERY_TYPE_TIMESTAMP, q + 1);
    g_launches.push_back(Launch{ h, gx, gy, gz, q });
    return r;
}

inline void *__cdecl QueryInterface(unsigned id)
{
    void *real = g_real_qi(id);
    if (std::find(g_ids.begin(), g_ids.end(), id) == g_ids.end()) g_ids.push_back(id);
    if (real == nullptr) return real;
    switch (id) {
    case kIdCreateCuFunction:
        g_real_create_cu_function = reinterpret_cast<PfnCreateCuFunction>(real);
        return reinterpret_cast<void *>(&CreateCuFunction);
    case kIdLaunchCuKernelChain:
        g_real_launch_chain = reinterpret_cast<PfnLaunchCuKernelChain>(real);
        return reinterpret_cast<void *>(&LaunchCuKernelChain);
    case kIdCreateWithName:
        g_real_create_named = reinterpret_cast<PfnCreateCubinWithName>(real);
        return reinterpret_cast<void *>(&CreateWithName);
    case kIdLaunch:
        g_real_launch = reinterpret_cast<PfnLaunchCubin>(real);
        return reinterpret_cast<void *>(&LaunchCubin);
    default:
        return real;
    }
}

inline FARPROC WINAPI GetProcAddressHook(HMODULE module, LPCSTR name)
{
    FARPROC real = GetProcAddress(module, name);
    if (real && HIWORD(reinterpret_cast<ULONG_PTR>(name)) != 0 && strcmp(name, "nvapi_QueryInterface") == 0) {
        g_real_qi = reinterpret_cast<PfnQueryInterface>(real);
        return reinterpret_cast<FARPROC>(&QueryInterface);
    }
    return real;
}

// 帧提交前: 把本帧的时间戳解析到回读缓冲
inline void Resolve(ID3D12GraphicsCommandList *list)
{
    if (g_on && g_heap && g_next) list->ResolveQueryData(g_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, g_next, g_readback, 0);
    if (g_on) TapResolve(list);
}

// fence 完成后: 累计本帧 (前 warmup 帧丢弃)
inline void Collect(ID3D12CommandQueue *queue, unsigned frame, unsigned warmup = 2)
{
    g_frame_idx = frame + 1;  // 下一帧的发射从这里开始计
    g_launch_seq = 0;
    if (g_on) TapCollect();
    if (!g_on || !g_next) return;
    UINT64 freq = 0;
    queue->GetTimestampFrequency(&freq);
    UINT64 *ts = nullptr;
    D3D12_RANGE range{ 0, g_next * sizeof(UINT64) };
    if (freq && SUCCEEDED(g_readback->Map(0, &range, reinterpret_cast<void **>(&ts)))) {
        if (frame >= warmup) {
            UINT64 first = ~0ULL, last = 0;
            for (const Launch &l : g_launches) {
                const Kernel &k = g_kernels[l.h];
                char key[64];
                sprintf_s(key, "%p|%u,%u,%u", l.h, l.gx, l.gy, l.gz);
                Acc &a = g_acc[key];
                a.ms += double(ts[l.q + 1] - ts[l.q]) * 1000.0 / double(freq);
                a.launches++;
                a.name = k.name;
                a.gx = l.gx; a.gy = l.gy; a.gz = l.gz;
                a.bx = k.bx; a.by = k.by; a.bz = k.bz; a.smem = k.smem;
                first = std::min(first, ts[l.q]);
                last = std::max(last, ts[l.q + 1]);
            }
            g_span_ms += double(last - first) * 1000.0 / double(freq);
            g_spans.push_back(double(last - first) * 1000.0 / double(freq));
            g_frames++;
        }
        D3D12_RANGE none{ 0, 0 };
        g_readback->Unmap(0, &none);
    }
    g_next = 0;
    g_launches.clear();
}

// 机器可读: 每行 ms/帧 \t 次/帧 \t grid \t block \t smem \t kernel 名 (完整 mangled 名，便于 c++filt)
inline void Report()
{
    if (!g_on) return;
    if (!g_frames) {
        std::printf("[prof] no data: QueryInterface %s, chain %s, kernels %zu, ids:", g_real_qi ? "hooked" : "missing",
            g_real_launch_chain ? "hooked" : "missing", g_kernels.size());
        for (unsigned id : g_ids) std::printf(" %08x", id);
        std::printf("\n");
        std::fflush(stdout);
        return;
    }
    std::vector<const Acc *> rows;
    double total = 0;
    for (auto &kv : g_acc) { rows.push_back(&kv.second); total += kv.second.ms; }
    std::sort(rows.begin(), rows.end(), [](const Acc *a, const Acc *b) { return a->ms > b->ms; });
    std::vector<double> s = g_spans;
    std::sort(s.begin(), s.end());
    std::printf("[prof] frames=%u span_ms=%.3f kernel_ms=%.3f distinct=%zu registered=%zu\n", g_frames,
        g_span_ms / g_frames, total / g_frames, rows.size(), g_kernels.size());
    std::printf("[prof] span_min=%.3f span_median=%.3f\n", s.front(), s[s.size() / 2]);
    for (const Acc *a : rows) {
        std::printf("[prof]\t%.4f\t%.1f\t%u,%u,%u\t%u,%u,%u\t%u\t%s\n", a->ms / g_frames,
            double(a->launches) / g_frames, a->gx, a->gy, a->gz, a->bx, a->by, a->bz, a->smem, a->name.c_str());
    }
    std::fflush(stdout);
}

inline void Install(HMODULE module, bool (*hook)(HMODULE, const char *, const char *, void *))
{
    const char *env = std::getenv("DLSS5_PROF");
    g_on = env && env[0] == '1';
    if (!g_on) return;
    if (const char *t = std::getenv("DLSS5_TRACE")) g_trace_frame = std::atoi(t);
    ParseTap();
    const bool ok = hook(module, "KERNEL32.dll", "GetProcAddress", reinterpret_cast<void *>(&GetProcAddressHook));
    std::printf("[prof] GetProcAddress hook %s\n", ok ? "installed" : "FAILED");
    std::atexit(Report);
}

}  // namespace prof
