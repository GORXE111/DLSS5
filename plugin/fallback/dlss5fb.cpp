// DLSS5 fallback mode: a dxgi.dll proxy for D3D12 games that ship no upscaler.
//
// The proxy forwards every dxgi export to the system dxgi.dll and patches the
// factory / swap chain vtables.  For each D3D12 swap chain it takes the finished
// frame at Present, runs DLSS-NR (feature 18, through nvngx_dlssnr.dll) on a
// downscaled copy, and writes  frame + upsample(NR(lo) - lo)  back into the
// back buffer before the real Present.  No depth; motion vectors come from the GPU's optical flow engine
// (NVIDIA Optical Flow, nvofapi64.dll from the driver) so DLSS-NR can keep its history; the UI is processed
// with the scene.
//
// NGX call sequence follows harness/nr-lab.cpp (derived from DLSS5 ReShade AIO,
// Apache-2.0): driver core _nvngx.dll Init -> snippet Init_Ext through the
// caller-identity bridge (dlss5_nvngx.dll) -> PopulateParameters -> CreateFeature(18).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>
#include <nvOpticalFlowD3D12.h>

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------- log / config

static wchar_t g_dir[MAX_PATH];          // directory of this dll, with trailing backslash
static FILE *g_log;
static std::mutex g_log_lock;

static void Log(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lock(g_log_lock);
    if (g_log == nullptr) {
        std::wstring path = std::wstring(g_dir) + L"dlss5fb.log";
        g_log = _wfopen(path.c_str(), L"w");
        if (g_log == nullptr) return;
    }
    SYSTEMTIME t;
    GetLocalTime(&t);
    std::fprintf(g_log, "%02d:%02d:%02d.%03d ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(g_log, fmt, args);
    va_end(args);
    std::fputc('\n', g_log);
    std::fflush(g_log);
}

struct Config {
    bool enabled = true;
    float scale = 0.5f;            // WorkingScale: NR runs at (W*scale, H*scale)
    float intensity = 1.0f;
    float local_tone = 1.0f;
    float local_structure = 1.0f;
    float skin_structure = -1.0f;
    int auto_mask = 1;
    int style = 0;
    int temporal = 1;              // 1: keep NR history, motion vectors from optical flow (0: every frame is a reset frame)
    int flow_perf = NV_OF_PERF_LEVEL_SLOW;     // optical flow quality: 5 slow/best (+1 ms vs medium at 540p, half the shimmer), 10 medium, 20 fast
    int flow_grid = 2;
    float mv_scale = 1.0f;         // research: multiplies the motion vectors
    float mv_const_x = 0.0f;       // research: non-zero = use this constant horizontal motion instead of optical flow
    int flow_filter = 1;           // 3x3 median of the flow vectors
    int flow_refine = 2;           // Lucas-Kanade iterations on top of the hardware flow (sub-pixel accuracy)
    int linear = 0;                // 1: feed DLSS-NR linear light (RGBA16F + IsHDR) instead of the sRGB picture
    float colour = 1.0f;           // ColourStrength: 0 = keep the game's colours (only DLSS5's brightness/detail), 1 = full
    float cut = 0.08f;             // scene cut: mean motion-compensated luma difference above this skips DLSS5 for that frame             // optical flow output grid (1, 2 or 4 pixels per vector, at NR resolution)
    float stabilize = 1.5f;        // NR input dead band in 1/255 steps: pixels that changed less keep their previous value
    float smooth = 0.2f;           // per-frame weight of a new NR change where the input did not change locally (1 = off)
    int compare = 0;               // 1: left half original, right half processed
    int toggle_key = VK_F10;
    int compare_key = VK_F11;
    int dump_frame = -1;           // write dlss5fb_in.ppm / dlss5fb_out.ppm at this frame (testing)
    int dump_count = 1;            // DumpCount > 1: also dlss5fb_out_<k>.ppm for the following frames (flicker tests)
};

static std::wstring IniPath() { return std::wstring(g_dir) + L"dlss5fb.ini"; }

static float IniFloat(const wchar_t *key, float def)
{
    wchar_t buf[64];
    GetPrivateProfileStringW(L"DLSS5", key, L"", buf, 64, IniPath().c_str());
    if (buf[0] == 0 || _wcsicmp(buf, L"auto") == 0) return def;
    return static_cast<float>(_wtof(buf));
}

static int IniInt(const wchar_t *key, int def)
{
    wchar_t buf[64];
    GetPrivateProfileStringW(L"DLSS5", key, L"", buf, 64, IniPath().c_str());
    if (buf[0] == 0 || _wcsicmp(buf, L"auto") == 0) return def;
    if (_wcsicmp(buf, L"true") == 0) return 1;
    if (_wcsicmp(buf, L"false") == 0) return 0;
    return static_cast<int>(wcstol(buf, nullptr, 0));
}

static Config ReadConfig()
{
    Config c;
    c.enabled = IniInt(L"Enabled", 1) != 0;
    c.scale = std::min(std::max(IniFloat(L"WorkingScale", c.scale), 0.25f), 1.0f);
    c.intensity = IniFloat(L"Intensity", c.intensity);
    c.local_tone = IniFloat(L"LocalTone", c.local_tone);
    c.local_structure = IniFloat(L"LocalStructure", c.local_structure);
    c.skin_structure = IniFloat(L"SkinStructure", c.skin_structure);
    c.auto_mask = IniInt(L"AutoMask", c.auto_mask);
    c.style = IniInt(L"Style", c.style);
    c.temporal = IniInt(L"Temporal", c.temporal);
    c.flow_perf = IniInt(L"FlowPerf", c.flow_perf);
    c.flow_grid = IniInt(L"FlowGrid", c.flow_grid);
    c.mv_scale = IniFloat(L"MvScale", c.mv_scale);
    c.mv_const_x = IniFloat(L"MvConstX", c.mv_const_x);
    c.flow_filter = IniInt(L"FlowFilter", c.flow_filter);
    c.flow_refine = IniInt(L"FlowRefine", c.flow_refine);
    c.linear = IniInt(L"Linear", c.linear);
    c.colour = std::min(std::max(IniFloat(L"ColourStrength", c.colour), 0.0f), 1.0f);
    c.cut = IniFloat(L"CutThreshold", c.cut);
    c.stabilize = std::max(0.0f, IniFloat(L"Stabilize", c.stabilize));
    c.smooth = std::min(std::max(IniFloat(L"Smooth", c.smooth), 0.02f), 1.0f);
    c.compare = IniInt(L"Compare", c.compare);
    c.toggle_key = IniInt(L"ToggleKey", c.toggle_key);
    c.compare_key = IniInt(L"CompareKey", c.compare_key);
    c.dump_frame = IniInt(L"DumpFrame", c.dump_frame);
    c.dump_count = std::max(1, IniInt(L"DumpCount", c.dump_count));
    wchar_t env[32];
    if (GetEnvironmentVariableW(L"DLSS5FB_DUMP", env, 32)) c.dump_frame = _wtoi(env);
    return c;
}

static FILETIME IniTime()
{
    WIN32_FILE_ATTRIBUTE_DATA a = {};
    GetFileAttributesExW(IniPath().c_str(), GetFileExInfoStandard, &a);
    return a.ftLastWriteTime;
}

// ---------------------------------------------------------------- NGX

static constexpr NVSDK_NGX_Feature kFeatureDlssNr = static_cast<NVSDK_NGX_Feature>(0x12);
static constexpr unsigned long long kGenericCustomCoreId = 0x0876232CULL;

using PfnCoreInit = NVSDK_NGX_Result (NVSDK_CONV *)(unsigned long long, const wchar_t *, ID3D12Device *, NVSDK_NGX_Version);
using PfnSnippetInit = NVSDK_NGX_Result (NVSDK_CONV *)(unsigned long long, const wchar_t *, ID3D12Device *,
    NVSDK_NGX_Version, const NVSDK_NGX_Parameter *);
using PfnGetParams = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Parameter **);
using PfnPopulate = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Parameter *);
using PfnCreate = NVSDK_NGX_Result (NVSDK_CONV *)(ID3D12GraphicsCommandList *, NVSDK_NGX_Feature,
    NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
using PfnEvaluate = NVSDK_NGX_Result (NVSDK_CONV *)(ID3D12GraphicsCommandList *, const NVSDK_NGX_Handle *,
    const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback_C);
using PfnRelease = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Handle *);
using PfnBridgeInit = NVSDK_NGX_Result (NVSDK_CONV *)(PfnSnippetInit, unsigned long long, const wchar_t *,
    ID3D12Device *, NVSDK_NGX_Version, const NVSDK_NGX_Parameter *);
using PfnBridgePopulate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnPopulate, NVSDK_NGX_Parameter *);
using PfnBridgeCreate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnCreate, ID3D12GraphicsCommandList *,
    NVSDK_NGX_Feature, NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
using PfnBridgeEvaluate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnEvaluate, ID3D12GraphicsCommandList *,
    const NVSDK_NGX_Handle *, const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback_C);
using PfnBridgeRelease = NVSDK_NGX_Result (NVSDK_CONV *)(PfnRelease, NVSDK_NGX_Handle *);

static struct {
    bool tried = false, ok = false;
    ID3D12Device *device = nullptr;
    NVSDK_NGX_Parameter *params = nullptr;
    PfnPopulate populate = nullptr;
    PfnCreate create = nullptr;
    PfnEvaluate evaluate = nullptr;
    PfnRelease release = nullptr;
    PfnBridgePopulate b_populate = nullptr;
    PfnBridgeCreate b_create = nullptr;
    PfnBridgeEvaluate b_evaluate = nullptr;
    PfnBridgeRelease b_release = nullptr;
} ngx;

static HMODULE LoadDriverCore()
{
    if (HMODULE m = GetModuleHandleW(L"_nvngx.dll")) return m;
    wchar_t sys[MAX_PATH];
    if (GetSystemDirectoryW(sys, MAX_PATH) == 0) return nullptr;
    std::wstring pattern = std::wstring(sys) + L"\\DriverStore\\FileRepository\\nv*.inf_amd64_*";
    WIN32_FIND_DATAW found = {};
    HANDLE search = FindFirstFileW(pattern.c_str(), &found);
    if (search == INVALID_HANDLE_VALUE) return nullptr;
    std::wstring best;
    FILETIME best_time = {};
    do {
        if ((found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) continue;
        std::wstring candidate = std::wstring(sys) + L"\\DriverStore\\FileRepository\\" + found.cFileName + L"\\_nvngx.dll";
        WIN32_FILE_ATTRIBUTE_DATA a = {};
        if (!GetFileAttributesExW(candidate.c_str(), GetFileExInfoStandard, &a)) continue;
        if (best.empty() || CompareFileTime(&a.ftLastWriteTime, &best_time) > 0) {
            best = candidate;
            best_time = a.ftLastWriteTime;
        }
    } while (FindNextFileW(search, &found));
    FindClose(search);
    return best.empty() ? nullptr : LoadLibraryExW(best.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}

static bool InitNgx(ID3D12Device *device)
{
    if (ngx.tried) return ngx.ok && ngx.device == device;
    ngx.tried = true;
    ngx.device = device;

    std::wstring snippet_path = std::wstring(g_dir) + L"nvngx_dlssnr.dll";
    std::wstring bridge_path = std::wstring(g_dir) + L"dlss5_nvngx.dll";
    HMODULE snippet = LoadLibraryExW(snippet_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE bridge = LoadLibraryExW(bridge_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE core = LoadDriverCore();
    Log("NGX: snippet=%p bridge=%p core=%p", snippet, bridge, core);
    if (snippet == nullptr || bridge == nullptr || core == nullptr) {
        Log("FAIL nvngx_dlssnr.dll / dlss5_nvngx.dll / driver _nvngx.dll not found (error %lu)", GetLastError());
        return false;
    }
    auto snippet_init = reinterpret_cast<PfnSnippetInit>(GetProcAddress(snippet, "NVSDK_NGX_D3D12_Init_Ext"));
    ngx.create = reinterpret_cast<PfnCreate>(GetProcAddress(snippet, "NVSDK_NGX_D3D12_CreateFeature"));
    ngx.evaluate = reinterpret_cast<PfnEvaluate>(GetProcAddress(snippet, "NVSDK_NGX_D3D12_EvaluateFeature"));
    ngx.release = reinterpret_cast<PfnRelease>(GetProcAddress(snippet, "NVSDK_NGX_D3D12_ReleaseFeature"));
    ngx.populate = reinterpret_cast<PfnPopulate>(GetProcAddress(snippet, "NVSDK_NGX_D3D12_PopulateParameters_Impl"));
    auto bridge_init = reinterpret_cast<PfnBridgeInit>(GetProcAddress(bridge, "NVNGXBridge_D3D12_InitExt"));
    ngx.b_populate = reinterpret_cast<PfnBridgePopulate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_PopulateParameters"));
    ngx.b_create = reinterpret_cast<PfnBridgeCreate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_CreateFeature"));
    ngx.b_evaluate = reinterpret_cast<PfnBridgeEvaluate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_EvaluateFeature"));
    ngx.b_release = reinterpret_cast<PfnBridgeRelease>(GetProcAddress(bridge, "NVNGXBridge_D3D12_ReleaseFeature"));
    auto core_init = reinterpret_cast<PfnCoreInit>(GetProcAddress(core, "NVSDK_NGX_D3D12_Init"));
    auto core_params = reinterpret_cast<PfnGetParams>(GetProcAddress(core, "NVSDK_NGX_D3D12_GetCapabilityParameters"));
    if (!snippet_init || !ngx.create || !ngx.evaluate || !ngx.release || !ngx.populate || !bridge_init ||
        !ngx.b_populate || !ngx.b_create || !ngx.b_evaluate || !ngx.b_release || !core_init || !core_params) {
        Log("FAIL missing NGX exports");
        return false;
    }

    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    std::wstring data = std::wstring(temp) + L"dlss5fb";
    CreateDirectoryW(data.c_str(), nullptr);
    NVSDK_NGX_Result r = core_init(kGenericCustomCoreId, data.c_str(), device, NVSDK_NGX_Version_API);
    Log("NGX core Init = 0x%08X", static_cast<unsigned>(r));
    if (NVSDK_NGX_FAILED(r)) return false;
    r = bridge_init(snippet_init, kGenericCustomCoreId, snippet_path.c_str(), device, NVSDK_NGX_Version_API, nullptr);
    Log("NGX snippet Init_Ext = 0x%08X", static_cast<unsigned>(r));
    if (NVSDK_NGX_FAILED(r)) return false;
    r = core_params(&ngx.params);
    Log("NGX GetCapabilityParameters = 0x%08X", static_cast<unsigned>(r));
    if (NVSDK_NGX_FAILED(r) || ngx.params == nullptr) return false;
    ngx.ok = true;
    return true;
}

static void SetControls(NVSDK_NGX_Parameter *p, const Config &c)
{
    p->Set("DLSSNR.Enabled", 1u);
    p->Set("DLSSNR.Hint.Render.Preset", 1);
    p->Set("DLSSNR.Style", static_cast<unsigned int>(c.style));
    p->Set("DLSSNR.Intensity", c.intensity);
    p->Set("DLSSNR.LocalToneStrength", c.local_tone);
    p->Set("DLSSNR.LocalStructureStrength", c.local_structure);
    p->Set("DLSSNR.SkinStructureStrength", c.skin_structure);
    p->Set("DLSSNR.UseAutoMask", static_cast<unsigned int>(c.auto_mask));
    p->Set("DLSSNR.UICorrection", 0u);
}

static void SetSizes(NVSDK_NGX_Parameter *p, UINT w, UINT h)
{
    p->Set("Width", w);
    p->Set("Height", h);
    p->Set("OutWidth", w);
    p->Set("OutHeight", h);
    p->Set("DLSSNR.InputWidth", w);
    p->Set("DLSSNR.InputHeight", h);
    p->Set("DLSSNR.Width", w);
    p->Set("DLSSNR.Height", h);
    p->Set("DLSSNR.OutputWidth", w);
    p->Set("DLSSNR.OutputHeight", h);
    p->Set("DLSSNR.Upscaling", 1u);
    p->Set("DLSSNR.ScalingRatio", 1.0f);
    p->Set("DLSSNR.Scale", 1.0f);
}

static NVSDK_NGX_Result SehCreate(ID3D12GraphicsCommandList *list, NVSDK_NGX_Handle **handle, DWORD *code)
{
    __try { return ngx.b_create(ngx.create, list, kFeatureDlssNr, ngx.params, handle); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return static_cast<NVSDK_NGX_Result>(0x7fffffff); }
}

static NVSDK_NGX_Result SehEvaluate(ID3D12GraphicsCommandList *list, NVSDK_NGX_Handle *handle, DWORD *code)
{
    __try { return ngx.b_evaluate(ngx.evaluate, list, handle, ngx.params, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return static_cast<NVSDK_NGX_Result>(0x7fffffff); }
}

// ---------------------------------------------------------------- optical flow (motion vectors)

static NV_OF_D3D12_API_FUNCTION_LIST g_of;

static bool LoadOpticalFlow()
{
    static int state = 0;   // 0 untried, 1 ok, -1 unavailable
    if (state) return state > 0;
    state = -1;
    HMODULE m = LoadLibraryW(L"nvofapi64.dll");
    auto create = m ? reinterpret_cast<NV_OF_STATUS (NVOFAPI *)(uint32_t, NV_OF_D3D12_API_FUNCTION_LIST *)>(
        GetProcAddress(m, "NvOFAPICreateInstanceD3D12")) : nullptr;
    if (create == nullptr) { Log("optical flow: nvofapi64.dll not available; no motion vectors"); return false; }
    NV_OF_STATUS st = create(NV_OF_API_VERSION, &g_of);
    if (st != NV_OF_SUCCESS) { Log("optical flow: NvOFAPICreateInstanceD3D12 = %d", st); return false; }
    state = 1;
    return true;
}

// ---------------------------------------------------------------- shaders

static const char kShader[] = R"(
Texture2D<float4> Src : register(t0);
Texture2D<float4> Lo  : register(t1);
Texture2D<float4> Nr  : register(t2);
Texture2D<float4> DeltaT : register(t3);
Texture2D<int2> Flow : register(t4);         // optical flow, S10.5 pixels, one vector per grid cell
Texture2D<float> Gray0 : register(t5);       // grey frames (which one is current: curGray)
Texture2D<float> Gray1 : register(t6);
RWTexture2D<float4> Out : register(u0);
RWTexture2D<float4> Stable : register(u1);   // rgb: last value fed to NR, a: how much it changed this frame
RWTexture2D<float4> Delta : register(u2);    // smoothed NR change (NR output - NR input)
// optical flow input (luma). ABGR8 "colour" input is not usable: the D3D12 driver path reads each byte of an RGBA8
// texel as a separate grey pixel (flow comes back 4x too large horizontally), measured with fbtest --pan
RWTexture2D<float> Gray : register(u3);
RWTexture2D<float2> Motion : register(u4);   // motion vectors for DLSS-NR (pixels, current -> previous)
RWBuffer<float> CutBuf : register(u5);       // [0] mean motion-compensated difference, [1] cut this frame, [2] cut last frame
SamplerState Lin : register(s0);
cbuffer C : register(b0) {
    float2 loSize; float2 bbSize; float taps; float direct; float split; float band;
    float smooth; float gain; float first; float flowValid;
    float grid; float mvScale; float mvConstX; float flowFilter;
    float refine; float curGray; float linearIO; float colour;
    float cutThreshold; float3 pad3;
};

float3 ToLinear(float3 c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float3 ToSrgb(float3 c) { c = max(c, 0); return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055; }
// NR input / output in the picture's own encoding (sRGB) or linear light (Linear=1)
float3 Encode(float3 c) { return linearIO > 0 ? ToSrgb(c) : c; }

[numthreads(8, 8, 1)]
void Down(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)loSize.x || id.y >= (uint)loSize.y) return;
    int n = (int)taps;
    float2 step = 1.0 / (loSize * n);
    float2 base = id.xy / loSize;
    float3 acc = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            acc += Src.SampleLevel(Lin, base + (float2(i, j) + 0.5) * step, 0).rgb;
    float3 x = saturate(acc / (n * n));
    Gray[id.xy] = dot(x, float3(0.299, 0.587, 0.114));
    // The network turns +-1/255 input noise (dither, film grain, GI noise) into visible flicker; without history
    // nothing averages it out. Feed the previous value while a pixel stays within the dead band.
    float3 s = Stable[id.xy].rgb;
    float3 d = abs(x - s);
    float m = max(d.r, max(d.g, d.b));
    if (band > 0 && m <= band) { x = s; m = 0; }
    Stable[id.xy] = float4(x, m);
    Out[id.xy] = float4(linearIO > 0 ? ToLinear(x) : x, 1);
}

// The network has global attention: a change anywhere (a HUD counter, one moving object) ripples through the
// whole output. Where the input did not change locally, move the NR change only part of the way per frame
// (exponential average); where it did change, take the new value at once, so moving content does not ghost.
[numthreads(8, 8, 1)]
void Smooth(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)loSize.x || id.y >= (uint)loSize.y) return;
    float m = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            m = max(m, Stable[clamp(int2(id.xy) + int2(dx, dy), 0, int2(loSize) - 1)].a);
    float3 dn = Encode(Nr.Load(int3(id.xy, 0)).rgb) - Encode(Lo.Load(int3(id.xy, 0)).rgb);
    // ColourStrength < 1: move towards the change's brightness part only, so the game keeps its own hues
    dn = lerp(dot(dn, float3(0.299, 0.587, 0.114)).xxx, dn, colour);
    float a = first > 0 || CutBuf[2] > 0 ? 1 : saturate(smooth + m * gain);   // right after a cut: no averaging
    if (CutBuf[1] > 0) { Delta[id.xy] = 0; return; }   // scene cut: show the game's frame, restart the average
    Delta[id.xy] = float4(lerp(Delta[id.xy].rgb, dn, a), 1);
}

float Median9(float v[9])
{
    // partial sorting network: min/max exchanges that leave the median in v[4]
    float t;
#define SX(a, b) t = min(v[a], v[b]); v[b] = max(v[a], v[b]); v[a] = t;
    SX(1, 2) SX(4, 5) SX(7, 8) SX(0, 1) SX(3, 4) SX(6, 7) SX(1, 2) SX(4, 5) SX(7, 8)
    SX(0, 3) SX(5, 8) SX(4, 7) SX(3, 6) SX(1, 4) SX(2, 5) SX(4, 7) SX(4, 2) SX(6, 4) SX(4, 2)
#undef SX
    return v[4];
}

// forward flow (current frame -> previous frame) is what DLSS-NR wants: history is sampled at pixel + mv.
// The flow is S10.5 fixed point (32 per pixel) at the NR resolution, one vector per grid cell.
[numthreads(8, 8, 1)]
void ToMotion(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)loSize.x || id.y >= (uint)loSize.y) return;
    if (mvConstX != 0) { Motion[id.xy] = float2(flowValid > 0 ? mvConstX : 0, 0); return; }
    if (flowValid <= 0) { Motion[id.xy] = 0; return; }
    int2 cell = int2(id.xy / (uint)grid);
    float2 f = float2(Flow.Load(int3(cell, 0)));
    if (flowFilter > 0) {
        uint fw, fh;
        Flow.GetDimensions(fw, fh);
        float vx[9], vy[9];
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                float2 q = float2(Flow.Load(int3(clamp(cell + int2(i - 1, j - 1), 0, int2(fw, fh) - 1), 0)));
                vx[j * 3 + i] = q.x;
                vy[j * 3 + i] = q.y;
            }
        f = float2(Median9(vx), Median9(vy));
    }
    float2 mv = f / 32.0;
    // Lucas-Kanade: the hardware flow is off by a few tenths of a pixel, smoothly in space, which DLSS-NR turns into
    // shimmer. Linearise the previous frame around x + mv and solve for the correction over a 5x5 window.
    float2 inv = 1.0 / loSize;
    for (int it = 0; it < (int)refine; ++it) {
        float a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
        for (int wy = -2; wy <= 2; ++wy)
            for (int wx = -2; wx <= 2; ++wx) {
                float2 pc = float2(id.xy) + float2(wx, wy) + 0.5;
                float2 pp = pc + mv;
                float ic = curGray < 0.5 ? Gray0.SampleLevel(Lin, pc * inv, 0) : Gray1.SampleLevel(Lin, pc * inv, 0);
                float ip, gx, gy;
                if (curGray < 0.5) {
                    ip = Gray1.SampleLevel(Lin, pp * inv, 0);
                    gx = 0.5 * (Gray1.SampleLevel(Lin, (pp + float2(1, 0)) * inv, 0) - Gray1.SampleLevel(Lin, (pp - float2(1, 0)) * inv, 0));
                    gy = 0.5 * (Gray1.SampleLevel(Lin, (pp + float2(0, 1)) * inv, 0) - Gray1.SampleLevel(Lin, (pp - float2(0, 1)) * inv, 0));
                } else {
                    ip = Gray0.SampleLevel(Lin, pp * inv, 0);
                    gx = 0.5 * (Gray0.SampleLevel(Lin, (pp + float2(1, 0)) * inv, 0) - Gray0.SampleLevel(Lin, (pp - float2(1, 0)) * inv, 0));
                    gy = 0.5 * (Gray0.SampleLevel(Lin, (pp + float2(0, 1)) * inv, 0) - Gray0.SampleLevel(Lin, (pp - float2(0, 1)) * inv, 0));
                }
                float r = ic - ip;
                a11 += gx * gx; a12 += gx * gy; a22 += gy * gy;
                b1 += gx * r; b2 += gy * r;
            }
        float det = a11 * a22 - a12 * a12;
        float tr = a11 + a22;
        // skip flat or one-directional (aperture) windows: smallest eigenvalue must be significant
        float lmin = 0.5 * (tr - sqrt(max(tr * tr - 4 * det, 0)));
        if (lmin < 1e-4) break;
        float2 d = float2(a22 * b1 - a12 * b2, a11 * b2 - a12 * b1) / det;
        mv += clamp(d, -1, 1);
    }
    Motion[id.xy] = mv * mvScale;
}

// Scene cut: warp the previous grey frame with the motion vectors and compare with this one. Normal motion leaves a
// small residual; after a cut nothing matches. DLSS-NR's own gate rejects the stale history only one frame late,
// so on a cut this frame shows the game's picture unchanged (Smooth writes a zero change).
groupshared float g_sum[256];
[numthreads(256, 1, 1)]
void Cut(uint3 tid : SV_GroupThreadID)
{
    uint n = (uint)loSize.x * (uint)loSize.y;
    float sum = 0, cnt = 0;
    float2 inv = 1.0 / loSize;
    for (uint i = tid.x * 7; i < n; i += 256 * 7) {
        uint2 p = uint2(i % (uint)loSize.x, i / (uint)loSize.x);
        float2 pc = float2(p) + 0.5;
        float2 pp = pc + (flowValid > 0 ? Motion[p] : 0);
        float ic = curGray < 0.5 ? Gray0.SampleLevel(Lin, pc * inv, 0) : Gray1.SampleLevel(Lin, pc * inv, 0);
        float ip = curGray < 0.5 ? Gray1.SampleLevel(Lin, pp * inv, 0) : Gray0.SampleLevel(Lin, pp * inv, 0);
        sum += abs(ic - ip);
        cnt += 1;
    }
    g_sum[tid.x] = sum / max(cnt, 1);
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid.x < s) g_sum[tid.x] += g_sum[tid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0) {
        float mean = g_sum[0] / 256;
        CutBuf[0] = mean;
        CutBuf[2] = CutBuf[1];
        CutBuf[1] = flowValid > 0 && cutThreshold > 0 && mean > cutThreshold ? 1 : 0;
    }
}

void VS(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD0)
{
    uv = float2((id << 1) & 2, id & 2);
    pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

float4 PS(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float4 b = Src.Load(int3(pos.xy, 0));
    if (CutBuf[1] > 0) return b;   // scene cut: this frame unchanged
    if (split > 0) {
        float mid = floor(bbSize.x * 0.5);
        if (abs(pos.x - 0.5 - mid) < 1) return float4(1, 1, 1, b.a);
        if (pos.x < mid) return b;
    }
    float3 c = direct > 0 && smooth >= 1 && colour >= 1 ? saturate(Encode(Nr.SampleLevel(Lin, uv, 0).rgb))
                                         : saturate(b.rgb + DeltaT.SampleLevel(Lin, uv, 0).rgb);
    return float4(c, b.a);
}
)";

using PfnD3DCompile = HRESULT (WINAPI *)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO *, ID3DInclude *,
    LPCSTR, LPCSTR, UINT, UINT, ID3DBlob **, ID3DBlob **);

static ID3DBlob *Compile(const char *entry, const char *target)
{
    static PfnD3DCompile compile = [] {
        HMODULE m = LoadLibraryW(L"d3dcompiler_47.dll");
        return m ? reinterpret_cast<PfnD3DCompile>(GetProcAddress(m, "D3DCompile")) : nullptr;
    }();
    if (compile == nullptr) { Log("FAIL d3dcompiler_47.dll"); return nullptr; }
    ID3DBlob *code = nullptr, *errors = nullptr;
    HRESULT hr = compile(kShader, sizeof(kShader) - 1, "dlss5fb", nullptr, nullptr, entry, target,
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(hr)) Log("FAIL shader %s: %s", entry, errors ? static_cast<const char *>(errors->GetBufferPointer()) : "?");
    if (errors) errors->Release();
    return code;
}

// ---------------------------------------------------------------- per swap chain state

template <class T> static void SafeRelease(T *&p) { if (p) { p->Release(); p = nullptr; } }

static DXGI_FORMAT LoFormat(bool linear) { return linear ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM; }
static const UINT kFrames = 3;
static const UINT kConstants = 24;   // root constants (cbuffer C)
// descriptor heap: SRV copy/lo/nr/delta/flow, then two UAV tables lo/stable/delta/gray/motion that differ only in
// which grey frame they write (the optical flow engine compares this frame's grey image with the previous one)
static const UINT kSrv = 0, kUav = 7, kUavCount = 6;

struct Chain {
    std::mutex lock;
    IDXGISwapChain3 *swapchain = nullptr;   // not owned
    ID3D12CommandQueue *queue = nullptr;
    ID3D12Device *device = nullptr;
    bool broken = false;
    DXGI_COLOR_SPACE_TYPE color_space = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;

    // slot i: first command list of frame i % kFrames; slot kFrames + i: the second one (after optical flow)
    ID3D12CommandAllocator *alloc[2 * kFrames] = {};
    UINT64 alloc_fence[2 * kFrames] = {};
    ID3D12GraphicsCommandList *list = nullptr, *list2 = nullptr;
    ID3D12Fence *fence = nullptr;
    UINT64 fence_value = 0;
    HANDLE event = nullptr;
    ID3D12RootSignature *root = nullptr;
    ID3D12PipelineState *down = nullptr;
    ID3D12PipelineState *smooth = nullptr;
    ID3D12PipelineState *to_motion = nullptr;
    ID3D12PipelineState *cut = nullptr;
    ID3D12Resource *cut_buf = nullptr;                 // 2 floats on the GPU
    ID3D12Resource *cut_rb[kFrames] = {};              // read back one frame late, for the log
    UINT64 cuts = 0;
    ID3D12PipelineState *combine = nullptr;
    DXGI_FORMAT combine_format = DXGI_FORMAT_UNKNOWN;
    ID3D12DescriptorHeap *heap = nullptr;
    ID3D12DescriptorHeap *rtv_heap = nullptr;

    UINT W = 0, H = 0, w = 0, h = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    ID3D12Resource *copy = nullptr, *lo = nullptr, *nr = nullptr, *depth = nullptr, *mv = nullptr, *stable = nullptr, *delta = nullptr;
    bool first_smooth = true;
    // optical flow session at NR resolution: two grey frames (current / previous) and the forward flow
    NvOFHandle of = nullptr;
    ID3D12Resource *gray[2] = {}, *flow = nullptr;
    NvOFGPUBufferHandle gray_h[2] = {}, flow_h = nullptr;
    ID3D12Fence *of_in = nullptr, *of_out = nullptr;
    UINT64 of_in_v = 0, of_out_v = 0;
    UINT grid = 1;
    int cur_gray = 0;
    bool have_prev = false;
    NVSDK_NGX_Handle *feature = nullptr;
    int feature_style = -1;
    int feature_linear = -1;

    Config cfg;
    FILETIME ini_time = {};
    DWORD last_ini_check = 0;
    bool key_down[2] = {};
    UINT64 frame = 0;
    UINT64 nr_frames = 0;
    UINT pend_W = 0, pend_H = 0;           // a new back buffer size waiting to settle (window being dragged)
    DWORD pend_since = 0;
    DXGI_FORMAT warned_format = DXGI_FORMAT_UNKNOWN;
    unsigned skip_logged = 0;              // skip reasons already written to the log (bit per reason)
};

// a frame was passed through unchanged: log each reason the first time it happens
static void Skip(Chain *c, unsigned bit, const char *why)
{
    if (c->skip_logged & bit) return;
    c->skip_logged |= bit;
    Log("frame %llu passed through: %s", c->frame, why);
}

static std::mutex g_chains_lock;
static std::unordered_map<IUnknown *, Chain *> g_chains;

static Chain *FindChain(IUnknown *swapchain)
{
    std::lock_guard<std::mutex> lock(g_chains_lock);
    auto it = g_chains.find(swapchain);
    return it == g_chains.end() ? nullptr : it->second;
}

static void WaitIdle(Chain *c)
{
    if (c->fence == nullptr) return;
    ++c->fence_value;
    c->queue->Signal(c->fence, c->fence_value);
    if (c->fence->GetCompletedValue() < c->fence_value) {
        c->fence->SetEventOnCompletion(c->fence_value, c->event);
        WaitForSingleObject(c->event, 5000);
    }
}

static void ReleaseFlow(Chain *c)
{
    // order as in NVIDIA's samples: unregister, release the textures, then destroy the session
    // (destroying first made the D3D12 driver read freed memory a moment later)
    if (c->of) {
        for (NvOFGPUBufferHandle *h : {&c->gray_h[0], &c->gray_h[1], &c->flow_h}) {
            if (*h == nullptr) continue;
            NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12 u = {*h};
            g_of.nvOFUnregisterResourceD3D12(&u);
            *h = nullptr;
        }
    }
    SafeRelease(c->gray[0]); SafeRelease(c->gray[1]); SafeRelease(c->flow);
    if (c->of) {
        g_of.nvOFDestroy(c->of);
        c->of = nullptr;
    }
    SafeRelease(c->of_in); SafeRelease(c->of_out);
    c->have_prev = false;
}

static void ReleaseSized(Chain *c)
{
    ReleaseFlow(c);
    if (c->feature) { ngx.b_release(ngx.release, c->feature); c->feature = nullptr; }
    SafeRelease(c->copy); SafeRelease(c->lo); SafeRelease(c->nr); SafeRelease(c->depth); SafeRelease(c->mv); SafeRelease(c->stable); SafeRelease(c->delta);
    c->W = c->H = c->w = c->h = 0;
}

static void DestroyChain(Chain *c)
{
    WaitIdle(c);
    ReleaseSized(c);
    for (auto &a : c->alloc) SafeRelease(a);
    SafeRelease(c->list); SafeRelease(c->list2); SafeRelease(c->fence); SafeRelease(c->root); SafeRelease(c->down); SafeRelease(c->smooth);
    SafeRelease(c->to_motion); SafeRelease(c->cut); SafeRelease(c->combine); SafeRelease(c->cut_buf);
    for (auto &r : c->cut_rb) SafeRelease(r);
    SafeRelease(c->heap); SafeRelease(c->rtv_heap); SafeRelease(c->queue); SafeRelease(c->device);
    if (c->event) CloseHandle(c->event);
    delete c;
}

static ID3D12Resource *MakeTexture(ID3D12Device *device, UINT w, UINT h, DXGI_FORMAT format, bool uav,
    D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource *r = nullptr;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
    if (FAILED(hr)) Log("FAIL CreateCommittedResource %ux%u fmt=%d: 0x%08X", w, h, format, static_cast<unsigned>(hr));
    return r;
}

static bool InitPipeline(Chain *c)
{
    ID3D12Device *d = c->device;
    for (auto &a : c->alloc)
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a)))) return false;
    if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c->alloc[0], nullptr, IID_PPV_ARGS(&c->list))) ||
        FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c->alloc[kFrames], nullptr, IID_PPV_ARGS(&c->list2))))
        return false;
    c->list->Close();
    c->list2->Close();
    if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c->fence)))) return false;
    c->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kUav, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE uav = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kUavCount, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &srv};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &uav};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants = {0, 0, kConstants};
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rs = {3, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ID3DBlob *blob = nullptr, *err = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        Log("FAIL root signature: %s", err ? static_cast<const char *>(err->GetBufferPointer()) : "?");
        return false;
    }
    HRESULT hr = d->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&c->root));
    blob->Release();
    if (FAILED(hr)) return false;

    ID3DBlob *cs = Compile("Down", "cs_5_0");
    if (cs == nullptr) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC cp = {};
    cp.pRootSignature = c->root;
    cp.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    hr = d->CreateComputePipelineState(&cp, IID_PPV_ARGS(&c->down));
    cs->Release();
    if (FAILED(hr)) return false;
    cs = Compile("Smooth", "cs_5_0");
    if (cs == nullptr) return false;
    cp.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    hr = d->CreateComputePipelineState(&cp, IID_PPV_ARGS(&c->smooth));
    cs->Release();
    if (FAILED(hr)) return false;
    cs = Compile("ToMotion", "cs_5_0");
    if (cs == nullptr) return false;
    cp.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    hr = d->CreateComputePipelineState(&cp, IID_PPV_ARGS(&c->to_motion));
    cs->Release();
    if (FAILED(hr)) return false;
    cs = Compile("Cut", "cs_5_0");
    if (cs == nullptr) return false;
    cp.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    hr = d->CreateComputePipelineState(&cp, IID_PPV_ARGS(&c->cut));
    cs->Release();
    if (FAILED(hr)) return false;
    {
        D3D12_HEAP_PROPERTIES dh = {D3D12_HEAP_TYPE_DEFAULT}, rh = {D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC bd = {};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = 256;
        bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(d->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                IID_PPV_ARGS(&c->cut_buf)))) return false;
        bd.Flags = D3D12_RESOURCE_FLAG_NONE;
        for (auto &r : c->cut_rb)
            if (FAILED(d->CreateCommittedResource(&rh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    IID_PPV_ARGS(&r)))) return false;
    }

    D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kUav + 2 * kUavCount,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    if (FAILED(d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&c->heap)))) return false;
    D3D12_DESCRIPTOR_HEAP_DESC rd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    return SUCCEEDED(d->CreateDescriptorHeap(&rd, IID_PPV_ARGS(&c->rtv_heap)));
}

static bool InitCombine(Chain *c, DXGI_FORMAT format)
{
    if (c->combine && c->combine_format == format) return true;
    SafeRelease(c->combine);
    ID3DBlob *vs = Compile("VS", "vs_5_0"), *ps = Compile("PS", "ps_5_0");
    if (vs == nullptr || ps == nullptr) return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gp = {};
    gp.pRootSignature = c->root;
    gp.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    gp.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    gp.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gp.SampleMask = UINT_MAX;
    gp.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gp.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gp.RasterizerState.DepthClipEnable = TRUE;
    gp.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gp.NumRenderTargets = 1;
    gp.RTVFormats[0] = format;
    gp.SampleDesc.Count = 1;
    HRESULT hr = c->device->CreateGraphicsPipelineState(&gp, IID_PPV_ARGS(&c->combine));
    vs->Release();
    ps->Release();
    c->combine_format = format;
    if (FAILED(hr)) Log("FAIL combine PSO fmt=%d: 0x%08X", format, static_cast<unsigned>(hr));
    return SUCCEEDED(hr);
}

static bool BeginList(Chain *c, UINT slot, ID3D12GraphicsCommandList *list = nullptr)
{
    if (list == nullptr) list = c->list;
    if (c->fence->GetCompletedValue() < c->alloc_fence[slot]) {
        c->fence->SetEventOnCompletion(c->alloc_fence[slot], c->event);
        WaitForSingleObject(c->event, 5000);
    }
    return SUCCEEDED(c->alloc[slot]->Reset()) && SUCCEEDED(list->Reset(c->alloc[slot], nullptr));
}

static void Submit(Chain *c, UINT slot, ID3D12GraphicsCommandList *list = nullptr)
{
    if (list == nullptr) list = c->list;
    list->Close();
    ID3D12CommandList *lists[] = {list};
    c->queue->ExecuteCommandLists(1, lists);
    c->alloc_fence[slot] = ++c->fence_value;
    c->queue->Signal(c->fence, c->fence_value);
}

static bool CreateFeature(Chain *c)
{
    if (c->feature) { ngx.b_release(ngx.release, c->feature); c->feature = nullptr; }
    ngx.params->Reset();
    ngx.b_populate(ngx.populate, ngx.params);
    ngx.params->Set("CreationNodeMask", 1u);
    ngx.params->Set("VisibilityNodeMask", 1u);
    ngx.params->Set("ResourceWidth", c->w);
    ngx.params->Set("ResourceHeight", c->h);
    ngx.params->Set("ResourceOutWidth", c->w);
    ngx.params->Set("ResourceOutHeight", c->h);
    ngx.params->Set("Output.Width", c->w);
    ngx.params->Set("Output.Height", c->h);
    ngx.params->Set("PerfQualityValue", static_cast<int>(NVSDK_NGX_PerfQuality_Value_UltraQuality));
    ngx.params->Set("DLSS.Feature.Create.Flags", static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
        NVSDK_NGX_DLSS_Feature_Flags_AutoExposure | NVSDK_NGX_DLSS_Feature_Flags_DepthInverted |
        (c->cfg.linear ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0)));
    ngx.params->Set("DLSS.Enable.Output.Subrects", 0);
    ngx.params->Set("DLSS.Denoise.Mode", 1);
    ngx.params->Set("DLSS.Roughness.Mode", 0u);
    ngx.params->Set("DLSS.Use.HW.Depth", 1u);
    SetSizes(ngx.params, c->w, c->h);
    SetControls(ngx.params, c->cfg);
    UINT slot = c->frame % kFrames;
    if (!BeginList(c, slot)) return false;
    DWORD code = 0;
    NVSDK_NGX_Result r = SehCreate(c->list, &c->feature, &code);
    Submit(c, slot);
    WaitIdle(c);
    Log("CreateFeature(18) %ux%u style=%d = 0x%08X seh=0x%08X handle=%p", c->w, c->h, c->cfg.style,
        static_cast<unsigned>(r), code, c->feature);
    c->feature_style = c->cfg.style;
    c->feature_linear = c->cfg.linear;
    return code == 0 && NVSDK_NGX_SUCCEED(r) && c->feature != nullptr;
}

static bool Supported(DXGI_FORMAT f)
{
    return f == DXGI_FORMAT_R8G8B8A8_UNORM || f == DXGI_FORMAT_B8G8R8A8_UNORM ||
        f == DXGI_FORMAT_R10G10B10A2_UNORM || f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ||
        f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
}

static bool RegisterOf(Chain *c, ID3D12Resource *r, NvOFGPUBufferHandle *h)
{
    NV_OF_REGISTER_RESOURCE_PARAMS_D3D12 rp = {};
    rp.resource = r;
    rp.inputFencePoint = {c->of_in, c->of_in_v};
    rp.hOFGpuBuffer = h;
    rp.outputFencePoint = {c->of_in, ++c->of_in_v};
    if (g_of.nvOFRegisterResourceD3D12(c->of, &rp) != NV_OF_SUCCESS) return false;
    c->of_in->SetEventOnCompletion(c->of_in_v, c->event);
    WaitForSingleObject(c->event, 5000);
    return true;
}

// optical flow at NR resolution; false = no motion vectors (every frame then is a reset frame)
static bool CreateFlow(Chain *c)
{
    if (!c->cfg.temporal || !LoadOpticalFlow()) return false;
    if (g_of.nvCreateOpticalFlowD3D12(c->device, &c->of) != NV_OF_SUCCESS || c->of == nullptr) {
        Log("optical flow: session creation failed");
        c->of = nullptr;
        return false;
    }
    uint32_t n = 0;
    std::vector<uint32_t> grids;
    if (g_of.nvOFGetCaps(c->of, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &n) == NV_OF_SUCCESS && n) {
        grids.resize(n);
        if (g_of.nvOFGetCaps(c->of, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, grids.data(), &n) != NV_OF_SUCCESS) grids.clear();
    }
    c->grid = 0;
    for (uint32_t g : grids)
        if (g >= static_cast<uint32_t>(c->cfg.flow_grid) && (c->grid == 0 || g < c->grid)) c->grid = g;
    if (c->grid == 0) c->grid = grids.empty() ? 4 : *std::max_element(grids.begin(), grids.end());
    NV_OF_INIT_PARAMS init = {};
    init.width = c->w;
    init.height = c->h;
    init.outGridSize = static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(c->grid);
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = static_cast<NV_OF_PERF_LEVEL>(c->cfg.flow_perf);
    init.enableExternalHints = NV_OF_FALSE;
    init.enableOutputCost = NV_OF_FALSE;
    init.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    init.enableRoi = NV_OF_FALSE;
    init.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
    init.enableGlobalFlow = NV_OF_FALSE;
    init.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    NV_OF_STATUS st = g_of.nvOFInit(c->of, &init);
    bool ok = st == NV_OF_SUCCESS &&
        SUCCEEDED(c->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c->of_in))) &&
        SUCCEEDED(c->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c->of_out)));
    if (ok) {
        c->of_in_v = c->of_out_v = 0;
        for (int i = 0; i < 2; ++i)
            c->gray[i] = MakeTexture(c->device, c->w, c->h, DXGI_FORMAT_R8_UNORM, true, D3D12_RESOURCE_STATE_COMMON);
        c->flow = MakeTexture(c->device, (c->w + c->grid - 1) / c->grid, (c->h + c->grid - 1) / c->grid,
            DXGI_FORMAT_R16G16_SINT, false, D3D12_RESOURCE_STATE_COMMON);
        ok = c->gray[0] && c->gray[1] && c->flow && RegisterOf(c, c->gray[0], &c->gray_h[0]) &&
            RegisterOf(c, c->gray[1], &c->gray_h[1]) && RegisterOf(c, c->flow, &c->flow_h);
    }
    if (!ok) {
        Log("optical flow: init failed (%d); no motion vectors", st);
        ReleaseFlow(c);
        return false;
    }
    c->cur_gray = 0;
    c->have_prev = false;
    Log("optical flow: %ux%u, grid %u, perf level %d", c->w, c->h, c->grid, c->cfg.flow_perf);
    return true;
}

// 1: ready, 0: skip this frame (size still changing), -1: failed
static int EnsureSized(Chain *c, const D3D12_RESOURCE_DESC &bb)
{
    UINT W = static_cast<UINT>(bb.Width), H = bb.Height;
    UINT w = std::max(64u, static_cast<UINT>(std::lround(W * c->cfg.scale)));
    UINT h = std::max(64u, static_cast<UINT>(std::lround(H * c->cfg.scale)));
    if (c->W == W && c->H == H && c->w == w && c->h == h && c->format == bb.Format && c->feature &&
        c->feature_style == c->cfg.style && c->feature_linear == c->cfg.linear)
        return 1;
    // creating the NR feature takes 0.3-1 s: while a window is being resized, pass frames through until
    // the size has stayed the same for 250 ms
    if (c->W != 0 && (c->W != W || c->H != H)) {
        DWORD now = GetTickCount();
        if (c->pend_W != W || c->pend_H != H) { c->pend_W = W; c->pend_H = H; c->pend_since = now; return 0; }
        if (now - c->pend_since < 250) return 0;
    }
    WaitIdle(c);
    ReleaseSized(c);
    c->W = W; c->H = H; c->w = w; c->h = h; c->format = bb.Format;
    Log("resize: back buffer %ux%u fmt=%d -> NR %ux%u (scale %.3f)", W, H, bb.Format, w, h, c->cfg.scale);
    c->copy = MakeTexture(c->device, W, H, bb.Format, false, D3D12_RESOURCE_STATE_COPY_DEST);
    const DXGI_FORMAT lo_format = LoFormat(c->cfg.linear != 0);
    c->lo = MakeTexture(c->device, w, h, lo_format, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c->nr = MakeTexture(c->device, w, h, lo_format, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c->depth = MakeTexture(c->device, w, h, DXGI_FORMAT_R32_FLOAT, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    c->mv = MakeTexture(c->device, w, h, DXGI_FORMAT_R16G16_FLOAT, true, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    c->stable = MakeTexture(c->device, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c->delta = MakeTexture(c->device, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c->first_smooth = true;
    if (!c->copy || !c->lo || !c->nr || !c->depth || !c->mv || !c->stable || !c->delta) return -1;
    CreateFlow(c);

    UINT inc = c->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE h0 = c->heap->GetCPUDescriptorHandleForHeapStart();
    ID3D12Resource *srvs[4] = {c->copy, c->lo, c->nr, c->delta};
    for (int i = 0; i < 4; ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = i == 0 ? bb.Format : i == 3 ? DXGI_FORMAT_R16G16B16A16_FLOAT : lo_format;
        if (sd.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        if (sd.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        c->device->CreateShaderResourceView(srvs[i], &sd, {h0.ptr + i * inc});
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC fd = {};   // flow (a null view when there is no optical flow)
    fd.Format = DXGI_FORMAT_R16G16_SINT;
    fd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    fd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    fd.Texture2D.MipLevels = 1;
    c->device->CreateShaderResourceView(c->flow, &fd, {h0.ptr + 4 * inc});
    fd.Format = DXGI_FORMAT_R8_UNORM;   // grey frames for the Lucas-Kanade refinement
    for (int t = 0; t < 2; ++t) c->device->CreateShaderResourceView(c->gray[t], &fd, {h0.ptr + (5 + t) * inc});
    for (UINT t = 0; t < 2; ++t) {
        const UINT u = kUav + t * kUavCount;
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        ud.Format = lo_format;
        c->device->CreateUnorderedAccessView(c->lo, nullptr, &ud, {h0.ptr + (u + 0) * inc});
        ud.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        c->device->CreateUnorderedAccessView(c->stable, nullptr, &ud, {h0.ptr + (u + 1) * inc});
        c->device->CreateUnorderedAccessView(c->delta, nullptr, &ud, {h0.ptr + (u + 2) * inc});
        ud.Format = DXGI_FORMAT_R8_UNORM;
        c->device->CreateUnorderedAccessView(c->gray[t], nullptr, &ud, {h0.ptr + (u + 3) * inc});
        ud.Format = DXGI_FORMAT_R16G16_FLOAT;
        c->device->CreateUnorderedAccessView(c->mv, nullptr, &ud, {h0.ptr + (u + 4) * inc});
        D3D12_UNORDERED_ACCESS_VIEW_DESC bd = {};
        bd.Format = DXGI_FORMAT_R32_FLOAT;
        bd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        bd.Buffer.NumElements = 3;
        c->device->CreateUnorderedAccessView(c->cut_buf, nullptr, &bd, {h0.ptr + (u + 5) * inc});
    }
    return InitCombine(c, bb.Format) && CreateFeature(c) ? 1 : -1;
}

static void Barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
{
    D3D12_RESOURCE_BARRIER x = {};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource = r;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a;
    x.Transition.StateAfter = b;
    list->ResourceBarrier(1, &x);
}

static void DumpPpm(Chain *c, ID3D12Resource *tex, D3D12_RESOURCE_STATES state, const wchar_t *name);

static const D3D12_RESOURCE_STATES kRead =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

static void ProcessFrame(Chain *c)
{
    ++c->frame;
    DWORD now = GetTickCount();
    if (now - c->last_ini_check > 1000) {
        c->last_ini_check = now;
        FILETIME t = IniTime();
        if (CompareFileTime(&t, &c->ini_time) != 0) {
            bool first = c->ini_time.dwLowDateTime == 0 && c->ini_time.dwHighDateTime == 0;
            c->ini_time = t;
            c->cfg = ReadConfig();
            Log("%s config: enabled=%d scale=%.3f intensity=%.2f tone=%.2f structure=%.2f skin=%.2f mask=%d style=%d temporal=%d",
                first ? "loaded" : "reloaded", c->cfg.enabled, c->cfg.scale, c->cfg.intensity, c->cfg.local_tone,
                c->cfg.local_structure, c->cfg.skin_structure, c->cfg.auto_mask, c->cfg.style, c->cfg.temporal);
        }
    }
    const int keys[2] = {c->cfg.toggle_key, c->cfg.compare_key};
    for (int i = 0; i < 2; ++i) {
        bool down = keys[i] > 0 && (GetAsyncKeyState(keys[i]) & 0x8000) != 0;
        if (down && !c->key_down[i]) {
            if (i == 0) c->cfg.enabled = !c->cfg.enabled;
            else c->cfg.compare = !c->cfg.compare;
            Log("hotkey: enabled=%d compare=%d", c->cfg.enabled, c->cfg.compare);
        }
        c->key_down[i] = down;
    }
    if (!c->cfg.enabled) { Skip(c, 1, "disabled (ini or F10)"); return; }
    if (c->broken) return;
    if (c->color_space != DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) { Skip(c, 2, "HDR colour space"); return; }

    UINT index = c->swapchain->GetCurrentBackBufferIndex();
    ID3D12Resource *bb = nullptr;
    if (FAILED(c->swapchain->GetBuffer(index, IID_PPV_ARGS(&bb)))) { Skip(c, 4, "GetBuffer failed"); return; }
    D3D12_RESOURCE_DESC desc = bb->GetDesc();
    if (!Supported(desc.Format) || desc.SampleDesc.Count != 1) {
        if (c->warned_format != desc.Format)
            Log("back buffer format %d / %u samples not supported (HDR or MSAA): frames passed through unchanged", desc.Format, desc.SampleDesc.Count);
        c->warned_format = desc.Format;
        bb->Release();
        return;
    }
    const int sized = EnsureSized(c, desc);
    if (sized <= 0) {
        if (sized == 0) Skip(c, 8, "back buffer size changing (waiting for it to settle)");
        if (sized < 0) {
            Log("FAIL creating resources / NR feature; fallback mode disabled");
            c->broken = true;
        }
        bb->Release();
        return;
    }

    const UINT slot = c->frame % kFrames;
    if (!BeginList(c, slot)) {
        Log("FAIL command list reset (device removed: 0x%08X); fallback mode disabled",
            static_cast<unsigned>(c->device->GetDeviceRemovedReason()));
        c->broken = true;
        bb->Release();
        return;
    }
    ID3D12GraphicsCommandList *l = c->list;
    Barrier(l, bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    l->CopyResource(c->copy, bb);
    Barrier(l, bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    Barrier(l, c->copy, D3D12_RESOURCE_STATE_COPY_DEST, kRead);

    // downscale (or convert at scale 1) into the NR input
    float taps = std::max(1.0f, std::round(0.5f / c->cfg.scale));
    const bool direct = c->w == c->W && c->h == c->H;
    float k[kConstants] = {float(c->w), float(c->h), float(c->W), float(c->H), direct ? 1.0f : taps,
        direct ? 1.0f : 0.0f, c->cfg.compare ? 1.0f : 0.0f, c->cfg.stabilize / 255.0f,
        c->cfg.smooth, 255.0f / 3.0f, c->first_smooth ? 1.0f : 0.0f, 0, float(c->grid), c->cfg.mv_scale, c->cfg.mv_const_x, c->cfg.flow_filter ? 1.0f : 0.0f,
        float(c->cfg.flow_refine), float(c->cur_gray), c->cfg.linear ? 1.0f : 0.0f, c->cfg.colour,
        c->cfg.cut, 0, 0, 0};
    c->first_smooth = false;
    const int cur = c->cur_gray;
    UINT inc = c->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE g0 = c->heap->GetGPUDescriptorHandleForHeapStart();
    const D3D12_GPU_DESCRIPTOR_HANDLE srv_table = {g0.ptr + kSrv * inc}, uav_table = {g0.ptr + (kUav + cur * kUavCount) * inc};
    if (c->of) Barrier(l, c->gray[cur], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    l->SetDescriptorHeaps(1, &c->heap);
    l->SetComputeRootSignature(c->root);
    l->SetPipelineState(c->down);
    l->SetComputeRootDescriptorTable(0, srv_table);
    l->SetComputeRootDescriptorTable(1, uav_table);
    l->SetComputeRoot32BitConstants(2, kConstants, k, 0);
    l->Dispatch((c->w + 7) / 8, (c->h + 7) / 8, 1);
    D3D12_RESOURCE_BARRIER ub = {};
    ub.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    ub.UAV.pResource = c->stable;
    l->ResourceBarrier(1, &ub);
    Barrier(l, c->lo, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kRead);

    // optical flow between this frame's grey image and the previous one: submit what we have, let the flow
    // engine wait for it, make the queue wait for the flow, and record the rest into the second command list
    UINT end_slot = slot;
    bool flow_valid = false;
    if (c->of) {
        Barrier(l, c->gray[cur], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        Submit(c, slot);
        c->queue->Signal(c->of_in, ++c->of_in_v);
        if (c->have_prev) {
            NV_OF_FENCE_POINT in = {c->of_in, c->of_in_v}, out = {c->of_out, ++c->of_out_v};
            NV_OF_EXECUTE_INPUT_PARAMS_D3D12 ip = {};
            ip.inputFrame = c->gray_h[cur];
            ip.referenceFrame = c->gray_h[1 - cur];
            ip.numFencePoints = 1;
            ip.fencePoint = &in;
            NV_OF_EXECUTE_OUTPUT_PARAMS_D3D12 op = {};
            op.outputBuffer = c->flow_h;
            op.fencePoint = &out;
            NV_OF_STATUS st = g_of.nvOFExecuteD3D12(c->of, &ip, &op);
            if (st == NV_OF_SUCCESS) {
                c->queue->Wait(c->of_out, c->of_out_v);
                flow_valid = true;
            } else {
                Skip(c, 16, "optical flow execute failed (reset frames until it works)");
            }
        }
        c->have_prev = true;
        c->cur_gray = 1 - cur;
        end_slot = kFrames + slot;
        if (!BeginList(c, end_slot, c->list2)) {
            Log("FAIL second command list reset; fallback mode disabled");
            c->broken = true;
            bb->Release();
            return;
        }
        l = c->list2;
    }
    k[11] = flow_valid ? 1.0f : 0.0f;

    // motion vectors for DLSS-NR (zeros without optical flow)
    l->SetDescriptorHeaps(1, &c->heap);
    l->SetComputeRootSignature(c->root);
    l->SetPipelineState(c->to_motion);
    l->SetComputeRootDescriptorTable(0, srv_table);
    l->SetComputeRootDescriptorTable(1, uav_table);
    l->SetComputeRoot32BitConstants(2, kConstants, k, 0);
    if (flow_valid) Barrier(l, c->flow, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(l, c->mv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    l->Dispatch((c->w + 7) / 8, (c->h + 7) / 8, 1);
    {
        D3D12_RESOURCE_BARRIER uv = {};
        uv.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uv.UAV.pResource = c->mv;
        l->ResourceBarrier(1, &uv);
        // scene cut check (also clears the flag when there is no flow this frame)
        l->SetPipelineState(c->cut);
        l->Dispatch(1, 1, 1);
        uv.UAV.pResource = c->cut_buf;
        l->ResourceBarrier(1, &uv);
        // log: read back what the GPU found the last time this slot was used (that frame has finished)
        const float *v = nullptr;
        D3D12_RANGE r = {0, 8};
        if (c->frame > kFrames && SUCCEEDED(c->cut_rb[slot]->Map(0, &r, reinterpret_cast<void **>(const_cast<float **>(&v))))) {
            if (v[1] > 0) {
                ++c->cuts;
                Log("scene cut: frame %llu, motion-compensated difference %.3f (DLSS5 skipped for that frame)", c->frame - kFrames, v[0]);
            }
            D3D12_RANGE none = {0, 0};
            c->cut_rb[slot]->Unmap(0, &none);
        }
        Barrier(l, c->cut_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        l->CopyBufferRegion(c->cut_rb[slot], 0, c->cut_buf, 0, 8);
        Barrier(l, c->cut_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    Barrier(l, c->mv, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (flow_valid) Barrier(l, c->flow, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);

    // DLSS-NR
    NVSDK_NGX_Parameter *p = ngx.params;
    // history only with real motion vectors (zero vectors would smear every moving edge)
    const bool reset = !c->cfg.temporal || !flow_valid;
    p->Set("Color", c->lo);
    p->Set("Output", c->nr);
    p->Set("Depth", c->depth);
    p->Set("MotionVectors", c->mv);
    p->Set("DLSSNR.Color", c->lo);
    p->Set("DLSSNR.Output", c->nr);
    p->Set("DLSSNR.Depth", c->depth);
    p->Set("DLSSNR.MVec", c->mv);
    p->Set("Reset", reset ? 1 : 0);
    p->Set("DLSSNR.Reset", reset ? 1 : 0);
    p->Set("Jitter.Offset.X", 0.0f);
    p->Set("Jitter.Offset.Y", 0.0f);
    p->Set("MV.Scale.X", 1.0f);
    p->Set("MV.Scale.Y", 1.0f);
    p->Set("DLSSNR.JitterOffsetX", 0.0f);
    p->Set("DLSSNR.JitterOffsetY", 0.0f);
    p->Set("DLSSNR.MVecScaleX", 1.0f);
    p->Set("DLSSNR.MVecScaleY", 1.0f);
    p->Set("DLSS.Pre.Exposure", 1.0f);
    p->Set("DLSS.Exposure.Scale", 1.0f);
    p->Set("DLSS.Render.Subrect.Dimensions.Width", c->w);
    p->Set("DLSS.Render.Subrect.Dimensions.Height", c->h);
    for (const char *name : {"Color", "MVec", "Depth", "Output"}) {
        char key[64];
        std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseX", name); p->Set(key, 0);
        std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseY", name); p->Set(key, 0);
        std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectWidth", name); p->Set(key, static_cast<int>(c->w));
        std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectHeight", name); p->Set(key, static_cast<int>(c->h));
    }
    p->Set("DLSSNR.DepthInverted", 1u);
    SetSizes(p, c->w, c->h);
    SetControls(p, c->cfg);
    DWORD code = 0;
    NVSDK_NGX_Result r = SehEvaluate(l, c->feature, &code);
    if (code != 0 || NVSDK_NGX_FAILED(r)) {
        Log("FAIL EvaluateFeature = 0x%08X seh=0x%08X; fallback mode disabled", static_cast<unsigned>(r), code);
        c->broken = true;
    }
    ++c->nr_frames;
    Barrier(l, c->nr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kRead);

    // smooth the NR change over time where the input did not change (NGX changed the heaps / root signature)
    l->SetDescriptorHeaps(1, &c->heap);
    l->SetComputeRootSignature(c->root);
    l->SetPipelineState(c->smooth);
    l->SetComputeRootDescriptorTable(0, srv_table);
    l->SetComputeRootDescriptorTable(1, uav_table);
    l->SetComputeRoot32BitConstants(2, kConstants, k, 0);
    l->Dispatch((c->w + 7) / 8, (c->h + 7) / 8, 1);
    Barrier(l, c->delta, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kRead);

    // combine into the back buffer (NGX changed the heaps / root signature)
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = c->rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_RENDER_TARGET_VIEW_DESC rd = {};
    rd.Format = desc.Format;
    rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    c->device->CreateRenderTargetView(bb, &rd, rtv);
    l->SetDescriptorHeaps(1, &c->heap);
    l->SetGraphicsRootSignature(c->root);
    l->SetPipelineState(c->combine);
    l->SetGraphicsRootDescriptorTable(0, srv_table);
    l->SetGraphicsRootDescriptorTable(1, uav_table);
    l->SetGraphicsRoot32BitConstants(2, kConstants, k, 0);
    l->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    D3D12_VIEWPORT vp = {0, 0, float(c->W), float(c->H), 0, 1};
    D3D12_RECT sc = {0, 0, LONG(c->W), LONG(c->H)};
    l->RSSetViewports(1, &vp);
    l->RSSetScissorRects(1, &sc);
    l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (!c->broken) l->DrawInstanced(3, 1, 0, 0);

    Barrier(l, bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    Barrier(l, c->copy, kRead, D3D12_RESOURCE_STATE_COPY_DEST);
    Barrier(l, c->lo, kRead, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Barrier(l, c->nr, kRead, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Barrier(l, c->delta, kRead, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Submit(c, end_slot, l);

    if (c->cfg.dump_frame >= 0 && c->frame == static_cast<UINT64>(c->cfg.dump_frame)) {
        WaitIdle(c);
        DumpPpm(c, c->copy, D3D12_RESOURCE_STATE_COPY_DEST, L"dlss5fb_in.ppm");
        DumpPpm(c, c->nr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"dlss5fb_nr.ppm");
        DumpPpm(c, bb, D3D12_RESOURCE_STATE_PRESENT, L"dlss5fb_out.ppm");
    }
    const long long dk = static_cast<long long>(c->frame) - c->cfg.dump_frame;
    if (c->cfg.dump_frame >= 0 && dk >= 1 && dk < c->cfg.dump_count) {
        WaitIdle(c);
        wchar_t name[64];
        swprintf_s(name, L"dlss5fb_in_%lld.ppm", dk);
        DumpPpm(c, c->copy, D3D12_RESOURCE_STATE_COPY_DEST, name);
        swprintf_s(name, L"dlss5fb_out_%lld.ppm", dk);
        DumpPpm(c, bb, D3D12_RESOURCE_STATE_PRESENT, name);
        swprintf_s(name, L"dlss5fb_lo_%lld.ppm", dk);
        DumpPpm(c, c->lo, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, name);
        swprintf_s(name, L"dlss5fb_nr_%lld.ppm", dk);
        DumpPpm(c, c->nr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, name);
        if (c->flow) {
            swprintf_s(name, L"dlss5fb_flow_%lld.bin", dk);
            DumpPpm(c, c->flow, D3D12_RESOURCE_STATE_COMMON, name);
            swprintf_s(name, L"dlss5fb_gray_%lld.bin", dk);
            DumpPpm(c, c->gray[1 - c->cur_gray], D3D12_RESOURCE_STATE_COMMON, name);
        }
    }
    if (c->frame == 1 || c->frame % 600 == 0) Log("frame %llu processed", c->frame);
    bb->Release();
}

static void DumpPpm(Chain *c, ID3D12Resource *tex, D3D12_RESOURCE_STATES state, const wchar_t *name)
{
    const bool raw = wcsstr(name, L".bin") != nullptr;   // raw texels, rows tightly packed (research dumps)
    D3D12_RESOURCE_DESC d = tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT64 total = 0;
    c->device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
    D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total;
    bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource *buf = nullptr;
    if (FAILED(c->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&buf)))) return;
    const UINT slot = (c->frame + 1) % kFrames;
    BeginList(c, slot);
    Barrier(c->list, tex, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst = {buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src = {tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    c->list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Barrier(c->list, tex, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    Submit(c, slot);
    WaitIdle(c);
    uint8_t *data = nullptr;
    buf->Map(0, nullptr, reinterpret_cast<void **>(&data));
    std::wstring path = std::wstring(g_dir) + name;
    if (raw) {
        if (FILE *f = _wfopen(path.c_str(), L"wb")) {
            const UINT bpp = d.Format == DXGI_FORMAT_R8_UNORM ? 1 : 4;
            for (UINT y = 0; y < d.Height; ++y)
                std::fwrite(data + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch, 1, static_cast<size_t>(d.Width) * bpp, f);
            std::fclose(f);
            Log("dumped %ls (%llux%u fmt %d)", name, d.Width, d.Height, d.Format);
        }
    } else if (FILE *f = _wfopen(path.c_str(), L"wb")) {
        std::fprintf(f, "P6\n%u %u\n255\n", static_cast<UINT>(d.Width), d.Height);
        const bool bgra = d.Format == DXGI_FORMAT_B8G8R8A8_UNORM || d.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        const bool rgb10 = d.Format == DXGI_FORMAT_R10G10B10A2_UNORM;
        std::vector<uint8_t> row(d.Width * 3);
        for (UINT y = 0; y < d.Height; ++y) {
            const uint8_t *s = data + fp.Offset + static_cast<size_t>(y) * fp.Footprint.RowPitch;
            for (UINT x = 0; x < d.Width; ++x) {
                uint8_t px[3];
                if (rgb10) {
                    uint32_t v = reinterpret_cast<const uint32_t *>(s)[x];
                    for (int k = 0; k < 3; ++k) px[k] = static_cast<uint8_t>((((v >> (10 * k)) & 1023) * 255 + 511) / 1023);
                } else {
                    px[0] = s[4 * x + (bgra ? 2 : 0)];
                    px[1] = s[4 * x + 1];
                    px[2] = s[4 * x + (bgra ? 0 : 2)];
                }
                std::memcpy(&row[3 * x], px, 3);
            }
            std::fwrite(row.data(), 1, row.size(), f);
        }
        std::fclose(f);
        Log("dumped %ls (%llux%u)", name, d.Width, d.Height);
    }
    buf->Unmap(0, nullptr);
    buf->Release();
}

// ---------------------------------------------------------------- vtable hooks

static void Patch(void **vtable, int index, void *hook, void **original)
{
    if (vtable[index] == hook) return;
    DWORD old;
    VirtualProtect(&vtable[index], sizeof(void *), PAGE_READWRITE, &old);
    if (*original == nullptr) *original = vtable[index];
    vtable[index] = hook;
    VirtualProtect(&vtable[index], sizeof(void *), old, &old);
}

using PfnPresent = HRESULT (STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
using PfnPresent1 = HRESULT (STDMETHODCALLTYPE *)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);
using PfnResize = HRESULT (STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT, UINT, DXGI_FORMAT, UINT);
using PfnResize1 = HRESULT (STDMETHODCALLTYPE *)(IDXGISwapChain3 *, UINT, UINT, UINT, DXGI_FORMAT, UINT,
    const UINT *, IUnknown *const *);
using PfnSetColorSpace = HRESULT (STDMETHODCALLTYPE *)(IDXGISwapChain3 *, DXGI_COLOR_SPACE_TYPE);
using PfnCreateSC = HRESULT (STDMETHODCALLTYPE *)(IDXGIFactory *, IUnknown *, DXGI_SWAP_CHAIN_DESC *, IDXGISwapChain **);
using PfnCreateSCHwnd = HRESULT (STDMETHODCALLTYPE *)(IDXGIFactory2 *, IUnknown *, HWND, const DXGI_SWAP_CHAIN_DESC1 *,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *, IDXGIOutput *, IDXGISwapChain1 **);
using PfnCreateSCCore = HRESULT (STDMETHODCALLTYPE *)(IDXGIFactory2 *, IUnknown *, IUnknown *,
    const DXGI_SWAP_CHAIN_DESC1 *, IDXGIOutput *, IDXGISwapChain1 **);
using PfnCreateSCComp = HRESULT (STDMETHODCALLTYPE *)(IDXGIFactory2 *, IUnknown *, const DXGI_SWAP_CHAIN_DESC1 *,
    IDXGIOutput *, IDXGISwapChain1 **);

static PfnPresent o_present;
static PfnPresent1 o_present1;
static PfnResize o_resize;
static PfnResize1 o_resize1;
static PfnSetColorSpace o_set_color_space;
static PfnCreateSC o_create_sc;
static PfnCreateSCHwnd o_create_sc_hwnd;
static PfnCreateSCCore o_create_sc_core;
static PfnCreateSCComp o_create_sc_comp;
static thread_local int t_depth;   // DXGI calls its own Present internally (Present -> Present1)

// Present statistics for the log: total / nested (DXGI calling itself) / DXGI_PRESENT_TEST / untracked swap chain
static std::atomic<unsigned long long> g_presents[4];
static std::atomic<DWORD> g_stats_time;

static void BeforePresent(IUnknown *sc, UINT flags)
{
    ++g_presents[0];
    DWORD now = GetTickCount(), last = g_stats_time.load();
    if (now - last > 10000 && g_stats_time.compare_exchange_strong(last, now))
        Log("presents: %llu total, %llu nested, %llu test-only, %llu on untracked swap chains", g_presents[0].load(),
            g_presents[1].load(), g_presents[2].load(), g_presents[3].load());
    if (t_depth++ != 0) { ++g_presents[1]; return; }
    if (flags & DXGI_PRESENT_TEST) { ++g_presents[2]; return; }
    Chain *c = FindChain(sc);
    if (c == nullptr) { ++g_presents[3]; return; }
    std::lock_guard<std::mutex> lock(c->lock);
    ProcessFrame(c);
}

static void AfterPresent(IUnknown *sc, HRESULT hr)
{
    --t_depth;
    if (hr != DXGI_ERROR_DEVICE_REMOVED && hr != DXGI_ERROR_DEVICE_RESET) return;
    if (Chain *c = FindChain(sc)) {
        std::lock_guard<std::mutex> lock(c->lock);
        if (!c->broken) Log("device removed (0x%08X); fallback mode disabled", static_cast<unsigned>(hr));
        c->broken = true;
    }
}

static HRESULT STDMETHODCALLTYPE HookPresent(IDXGISwapChain *sc, UINT sync, UINT flags)
{
    BeforePresent(sc, flags);
    HRESULT hr = o_present(sc, sync, flags);
    AfterPresent(sc, hr);
    return hr;
}

static HRESULT STDMETHODCALLTYPE HookPresent1(IDXGISwapChain1 *sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS *pp)
{
    BeforePresent(sc, flags);
    HRESULT hr = o_present1(sc, sync, flags, pp);
    AfterPresent(sc, hr);
    return hr;
}

// we hold no back buffer references, so ResizeBuffers only needs our queued work to finish;
// the sized resources are rebuilt (debounced) on a later Present
static void BeforeResize(IUnknown *sc)
{
    if (Chain *c = FindChain(sc)) {
        std::lock_guard<std::mutex> lock(c->lock);
        WaitIdle(c);
    }
}

using PfnUnkRelease = ULONG (STDMETHODCALLTYPE *)(IUnknown *);
static PfnUnkRelease o_release;

// last reference gone: drop our state for this swap chain (the address may be reused by the next one)
static ULONG STDMETHODCALLTYPE HookRelease(IUnknown *sc)
{
    ULONG left = o_release(sc);
    if (left == 0) {
        Chain *c = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_chains_lock);
            auto it = g_chains.find(sc);
            if (it != g_chains.end()) { c = it->second; g_chains.erase(it); }
        }
        if (c) {
            Log("swap chain %p released (%llu frames)", sc, c->frame);
            DestroyChain(c);
        }
    }
    return left;
}

static HRESULT STDMETHODCALLTYPE HookResize(IDXGISwapChain *sc, UINT n, UINT w, UINT h, DXGI_FORMAT f, UINT flags)
{
    BeforeResize(sc);
    return o_resize(sc, n, w, h, f, flags);
}

static HRESULT STDMETHODCALLTYPE HookResize1(IDXGISwapChain3 *sc, UINT n, UINT w, UINT h, DXGI_FORMAT f, UINT flags,
    const UINT *masks, IUnknown *const *queues)
{
    BeforeResize(sc);
    return o_resize1(sc, n, w, h, f, flags, masks, queues);
}

static HRESULT STDMETHODCALLTYPE HookSetColorSpace(IDXGISwapChain3 *sc, DXGI_COLOR_SPACE_TYPE cs)
{
    if (Chain *c = FindChain(sc)) {
        std::lock_guard<std::mutex> lock(c->lock);
        if (cs != c->color_space) Log("color space %d%s", cs, cs == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 ? "" : " (HDR: not processed)");
        c->color_space = cs;
    }
    return o_set_color_space(sc, cs);
}

static std::string ModuleOf(void *addr)
{
    HMODULE m = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            static_cast<LPCSTR>(addr), &m))
        GetModuleFileNameA(m, name, MAX_PATH);
    const char *base = strrchr(name, '\\');
    return base ? base + 1 : name;
}

// diagnostics: report when someone else rewrites our swap chain vtable entries
static void **g_sc_vtable;
static DWORD WINAPI Watchdog(LPVOID)
{
    const int idx[2] = {8, 22};
    void *const mine[2] = {reinterpret_cast<void *>(&HookPresent), reinterpret_cast<void *>(&HookPresent1)};
    void *seen[2] = {mine[0], mine[1]};
    for (;;) {
        Sleep(500);
        for (int i = 0; i < 2; ++i) {
            void *now = g_sc_vtable[idx[i]];
            if (now != seen[i]) {
                Log("vtable[%d] changed: %p (%s) -> %p (%s)", idx[i], seen[i], ModuleOf(seen[i]).c_str(), now, ModuleOf(now).c_str());
                seen[i] = now;
            }
        }
    }
}

static void Track(IUnknown *device, IUnknown *swapchain)
{
    if (device == nullptr || swapchain == nullptr) return;
    ID3D12CommandQueue *queue = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&queue)))) return;   // D3D11 etc.: pass through
    IDXGISwapChain3 *sc3 = nullptr;
    if (FAILED(swapchain->QueryInterface(IID_PPV_ARGS(&sc3)))) { queue->Release(); return; }
    sc3->Release();   // keep a weak pointer; the swap chain owns itself

    void **vt = *reinterpret_cast<void ***>(sc3);
    Patch(vt, 2, reinterpret_cast<void *>(&HookRelease), reinterpret_cast<void **>(&o_release));
    Patch(vt, 8, reinterpret_cast<void *>(&HookPresent), reinterpret_cast<void **>(&o_present));
    Patch(vt, 13, reinterpret_cast<void *>(&HookResize), reinterpret_cast<void **>(&o_resize));
    Patch(vt, 22, reinterpret_cast<void *>(&HookPresent1), reinterpret_cast<void **>(&o_present1));
    Patch(vt, 38, reinterpret_cast<void *>(&HookSetColorSpace), reinterpret_cast<void **>(&o_set_color_space));
    Patch(vt, 39, reinterpret_cast<void *>(&HookResize1), reinterpret_cast<void **>(&o_resize1));
    if (g_sc_vtable == nullptr) {
        g_sc_vtable = vt;
        CloseHandle(CreateThread(nullptr, 0, Watchdog, nullptr, 0, nullptr));
    }

    auto *c = new Chain;
    c->swapchain = sc3;
    c->queue = queue;
    queue->GetDevice(IID_PPV_ARGS(&c->device));
    D3D12_COMMAND_QUEUE_DESC qd = queue->GetDesc();
    bool ok = qd.Type == D3D12_COMMAND_LIST_TYPE_DIRECT && InitNgx(c->device) && InitPipeline(c);
    if (!ok) {
        Log("swap chain %p: setup failed (queue type %d), passing through", swapchain, qd.Type);
        c->broken = true;
    } else {
        Log("swap chain %p tracked (D3D12 queue %p)", swapchain, queue);
    }
    std::lock_guard<std::mutex> lock(g_chains_lock);
    auto it = g_chains.find(swapchain);
    if (it != g_chains.end()) it->second->broken = true;   // stale entry at a reused address: leak it, never touch it
    g_chains[swapchain] = c;
}

static HRESULT STDMETHODCALLTYPE HookCreateSC(IDXGIFactory *f, IUnknown *dev, DXGI_SWAP_CHAIN_DESC *desc, IDXGISwapChain **out)
{
    HRESULT hr = o_create_sc(f, dev, desc, out);
    if (SUCCEEDED(hr) && out) Track(dev, *out);
    return hr;
}

static HRESULT STDMETHODCALLTYPE HookCreateSCHwnd(IDXGIFactory2 *f, IUnknown *dev, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1 *desc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *fs, IDXGIOutput *output, IDXGISwapChain1 **out)
{
    HRESULT hr = o_create_sc_hwnd(f, dev, hwnd, desc, fs, output, out);
    if (SUCCEEDED(hr) && out) Track(dev, *out);
    return hr;
}

static HRESULT STDMETHODCALLTYPE HookCreateSCCore(IDXGIFactory2 *f, IUnknown *dev, IUnknown *window,
    const DXGI_SWAP_CHAIN_DESC1 *desc, IDXGIOutput *output, IDXGISwapChain1 **out)
{
    HRESULT hr = o_create_sc_core(f, dev, window, desc, output, out);
    if (SUCCEEDED(hr) && out) Track(dev, *out);
    return hr;
}

static HRESULT STDMETHODCALLTYPE HookCreateSCComp(IDXGIFactory2 *f, IUnknown *dev, const DXGI_SWAP_CHAIN_DESC1 *desc,
    IDXGIOutput *output, IDXGISwapChain1 **out)
{
    HRESULT hr = o_create_sc_comp(f, dev, desc, output, out);
    if (SUCCEEDED(hr) && out) Track(dev, *out);
    return hr;
}

static void HookFactory(void *factory)
{
    if (factory == nullptr) return;
    IUnknown *unk = static_cast<IUnknown *>(factory);
    IDXGIFactory2 *f2 = nullptr;
    if (FAILED(unk->QueryInterface(IID_PPV_ARGS(&f2)))) return;
    void **vt = *reinterpret_cast<void ***>(f2);
    Patch(vt, 10, reinterpret_cast<void *>(&HookCreateSC), reinterpret_cast<void **>(&o_create_sc));
    Patch(vt, 15, reinterpret_cast<void *>(&HookCreateSCHwnd), reinterpret_cast<void **>(&o_create_sc_hwnd));
    Patch(vt, 16, reinterpret_cast<void *>(&HookCreateSCCore), reinterpret_cast<void **>(&o_create_sc_core));
    Patch(vt, 24, reinterpret_cast<void *>(&HookCreateSCComp), reinterpret_cast<void **>(&o_create_sc_comp));
    f2->Release();
}

// ---------------------------------------------------------------- dxgi exports

extern "C" void *g_real[20];   // filled for the assembly forwarders in exports.asm
void *g_real[20];
static const char *const kExports[20] = {
    "ApplyCompatResolutionQuirking", "CompatString", "CompatValue", "CreateDXGIFactory", "CreateDXGIFactory1",
    "CreateDXGIFactory2", "DXGID3D10CreateDevice", "DXGID3D10CreateLayeredDevice", "DXGID3D10GetLayeredDeviceSize",
    "DXGID3D10RegisterLayers", "DXGIDeclareAdapterRemovalSupport", "DXGIDisableVBlankVirtualization",
    "DXGIDumpJournal", "DXGIGetDebugInterface1", "DXGIReportAdapterConfiguration", "PIXBeginCapture",
    "PIXEndCapture", "PIXGetCaptureState", "SetAppCompatStringPointer", "UpdateHMDEmulationStatus"};

static void LoadRealDxgi()
{
    wchar_t sys[MAX_PATH];
    GetSystemDirectoryW(sys, MAX_PATH);
    HMODULE real = LoadLibraryW((std::wstring(sys) + L"\\dxgi.dll").c_str());
    for (int i = 0; i < 20; ++i) g_real[i] = real ? reinterpret_cast<void *>(GetProcAddress(real, kExports[i])) : nullptr;
    Log("dlss5fb loaded into %ls; system dxgi=%p", [] {
        static wchar_t exe[MAX_PATH];
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        return exe;
    }(), real);
}

extern "C" HRESULT WINAPI Proxy_CreateDXGIFactory(REFIID riid, void **out)
{
    HRESULT hr = reinterpret_cast<HRESULT (WINAPI *)(REFIID, void **)>(g_real[3])(riid, out);
    if (SUCCEEDED(hr)) HookFactory(*out);
    return hr;
}

extern "C" HRESULT WINAPI Proxy_CreateDXGIFactory1(REFIID riid, void **out)
{
    HRESULT hr = reinterpret_cast<HRESULT (WINAPI *)(REFIID, void **)>(g_real[4])(riid, out);
    if (SUCCEEDED(hr)) HookFactory(*out);
    return hr;
}

extern "C" HRESULT WINAPI Proxy_CreateDXGIFactory2(UINT flags, REFIID riid, void **out)
{
    HRESULT hr = reinterpret_cast<HRESULT (WINAPI *)(UINT, REFIID, void **)>(g_real[5])(flags, riid, out);
    if (SUCCEEDED(hr)) HookFactory(*out);
    return hr;
}

// diagnostics: log access violations (module + offset); the process then handles or crashes as it would anyway
static LONG CALLBACK CrashLogger(EXCEPTION_POINTERS *e)
{
    static std::atomic<int> logged{0};
    if (e->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && logged++ < 3) {
        void *at = e->ExceptionRecord->ExceptionAddress;
        HMODULE m = nullptr;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            static_cast<LPCSTR>(at), &m);
        Log("access violation at %s+0x%llx (%s address %p), thread %lu", ModuleOf(at).c_str(),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(at) - reinterpret_cast<uintptr_t>(m)),
            e->ExceptionRecord->ExceptionInformation[0] ? "writing" : "reading",
            reinterpret_cast<void *>(e->ExceptionRecord->ExceptionInformation[1]), GetCurrentThreadId());
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
        GetModuleFileNameW(module, g_dir, MAX_PATH);
        if (wchar_t *slash = wcsrchr(g_dir, L'\\')) slash[1] = 0;
        LoadRealDxgi();
        AddVectoredExceptionHandler(0, CrashLogger);
    }
    return TRUE;
}
