// nr11: does DLSS-NR work through its D3D11 entry points?
// Creates a D3D11 device, initialises NGX (driver core + nvngx_dlssnr.dll through the caller-identity bridge),
// runs N frames on the same synthetic pattern nr-lab uses and writes the last output as a PPM.
//
//   nr11.exe [--dir <folder with nvngx_dlssnr.dll and nvngx.dll>] [--size WxH] [--frames N] [--out file.ppm]
//
// Build: see build.bat.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr NVSDK_NGX_Feature kFeatureDlssNr = static_cast<NVSDK_NGX_Feature>(0x12);
static constexpr unsigned long long kGenericCustomCoreId = 0x0876232CULL;

using PfnCoreInit = NVSDK_NGX_Result (NVSDK_CONV *)(unsigned long long, const wchar_t *, ID3D11Device *, NVSDK_NGX_Version);
using PfnSnippetInit = NVSDK_NGX_Result (NVSDK_CONV *)(unsigned long long, const wchar_t *, ID3D11Device *,
    NVSDK_NGX_Version, const NVSDK_NGX_Parameter *);
using PfnGetParams = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Parameter **);
using PfnPopulate = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Parameter *);
using PfnCreate = NVSDK_NGX_Result (NVSDK_CONV *)(ID3D11DeviceContext *, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *,
    NVSDK_NGX_Handle **);
using PfnEvaluate = NVSDK_NGX_Result (NVSDK_CONV *)(ID3D11DeviceContext *, const NVSDK_NGX_Handle *,
    const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback_C);
using PfnRelease = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Handle *);
// The bridge only forwards pointers, so its D3D12-named entry points work for D3D11 objects too.
using PfnBridgeInit = NVSDK_NGX_Result (NVSDK_CONV *)(PfnSnippetInit, unsigned long long, const wchar_t *,
    ID3D11Device *, NVSDK_NGX_Version, const NVSDK_NGX_Parameter *);
using PfnBridgePopulate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnPopulate, NVSDK_NGX_Parameter *);
using PfnBridgeCreate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnCreate, ID3D11DeviceContext *, NVSDK_NGX_Feature,
    NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
using PfnBridgeEvaluate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnEvaluate, ID3D11DeviceContext *,
    const NVSDK_NGX_Handle *, const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback_C);

static HMODULE LoadDriverCore()
{
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

// nr-lab's MakeColorPattern (sRGB, R8G8B8A8_UNORM)
static std::vector<unsigned char> Pattern(UINT w, UINT h)
{
    std::vector<unsigned char> p(static_cast<size_t>(w) * h * 4);
    for (UINT y = 0; y < h; ++y) {
        for (UINT x = 0; x < w; ++x) {
            const bool checker = (((x / 12) ^ (y / 12)) & 1) != 0;
            const bool line = (x % 61 < 2) || (y % 47 < 2) || ((x + y) % 79 < 2);
            const float r = line ? 1.0f : (checker ? 0.82f : 0.06f);
            const float g = line ? 0.18f : (checker ? 0.11f : 0.68f);
            const float b = line ? 0.04f : (checker ? 0.55f : 0.09f);
            unsigned char *q = &p[(static_cast<size_t>(y) * w + x) * 4];
            q[0] = static_cast<unsigned char>(r * 255.0f + 0.5f);
            q[1] = static_cast<unsigned char>(g * 255.0f + 0.5f);
            q[2] = static_cast<unsigned char>(b * 255.0f + 0.5f);
            q[3] = 255;
        }
    }
    return p;
}

static ID3D11Texture2D *MakeTexture(ID3D11Device *dev, UINT w, UINT h, DXGI_FORMAT fmt, UINT bind, const void *data, UINT pitch)
{
    D3D11_TEXTURE2D_DESC d = {};
    d.Width = w; d.Height = h; d.MipLevels = 1; d.ArraySize = 1; d.Format = fmt;
    d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = bind;
    D3D11_SUBRESOURCE_DATA init = {data, pitch, 0};
    ID3D11Texture2D *t = nullptr;
    HRESULT hr = dev->CreateTexture2D(&d, data ? &init : nullptr, &t);
    if (FAILED(hr)) std::printf("CreateTexture2D fmt=%d failed 0x%08lX\n", fmt, hr);
    return t;
}

static NVSDK_NGX_Result SehCreate(PfnBridgeCreate b, PfnCreate f, ID3D11DeviceContext *ctx, NVSDK_NGX_Parameter *p,
    NVSDK_NGX_Handle **h, DWORD *code)
{
    __try { return b(f, ctx, kFeatureDlssNr, p, h); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return static_cast<NVSDK_NGX_Result>(0x7fffffff); }
}

static NVSDK_NGX_Result SehEvaluate(PfnBridgeEvaluate b, PfnEvaluate f, ID3D11DeviceContext *ctx, NVSDK_NGX_Handle *h,
    NVSDK_NGX_Parameter *p, DWORD *code)
{
    __try { return b(f, ctx, h, p, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return static_cast<NVSDK_NGX_Result>(0x7fffffff); }
}

static void SetSizes(NVSDK_NGX_Parameter *p, UINT w, UINT h)
{
    p->Set("Width", w); p->Set("Height", h); p->Set("OutWidth", w); p->Set("OutHeight", h);
    p->Set("DLSSNR.InputWidth", w); p->Set("DLSSNR.InputHeight", h);
    p->Set("DLSSNR.Width", w); p->Set("DLSSNR.Height", h);
    p->Set("DLSSNR.OutputWidth", w); p->Set("DLSSNR.OutputHeight", h);
    p->Set("DLSSNR.Upscaling", 1u); p->Set("DLSSNR.ScalingRatio", 1.0f); p->Set("DLSSNR.Scale", 1.0f);
}

static void SetControls(NVSDK_NGX_Parameter *p)
{
    p->Set("DLSSNR.Enabled", 1u);
    p->Set("DLSSNR.Hint.Render.Preset", 1);
    p->Set("DLSSNR.Style", 0u);
    p->Set("DLSSNR.Intensity", 1.0f);
    p->Set("DLSSNR.LocalToneStrength", 1.0f);
    p->Set("DLSSNR.LocalStructureStrength", 1.0f);
    p->Set("DLSSNR.SkinStructureStrength", -1.0f);
    p->Set("DLSSNR.UseAutoMask", 1u);
    p->Set("DLSSNR.UICorrection", 0u);
}

int wmain(int argc, wchar_t **argv)
{
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir(exe);
    dir.erase(dir.find_last_of(L'\\') + 1);
    std::wstring out = L"nr11-output.ppm";
    UINT w = 640, h = 360;
    int frames = 1;
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--dir" && i + 1 < argc) { dir = argv[++i]; if (dir.back() != L'\\') dir += L'\\'; }
        else if (a == L"--size" && i + 1 < argc) swscanf_s(argv[++i], L"%ux%u", &w, &h);
        else if (a == L"--frames" && i + 1 < argc) frames = _wtoi(argv[++i]);
        else if (a == L"--out" && i + 1 < argc) out = argv[++i];
    }

    ID3D11Device *dev = nullptr;
    ID3D11DeviceContext *ctx = nullptr;
    D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0, want[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, want, 2, D3D11_SDK_VERSION, &dev, &level, &ctx);
    std::printf("D3D11CreateDevice = 0x%08lX level=0x%X\n", hr, level);
    if (FAILED(hr)) return 1;

    std::wstring snippet_path = dir + L"nvngx_dlssnr.dll", bridge_path = dir + L"nvngx.dll";
    HMODULE snippet = LoadLibraryExW(snippet_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE bridge = LoadLibraryExW(bridge_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE core = LoadDriverCore();
    std::printf("modules: snippet=%p bridge=%p core=%p\n", snippet, bridge, core);
    if (!snippet || !bridge || !core) return 1;
    auto snippet_init = reinterpret_cast<PfnSnippetInit>(GetProcAddress(snippet, "NVSDK_NGX_D3D11_Init_Ext"));
    auto create = reinterpret_cast<PfnCreate>(GetProcAddress(snippet, "NVSDK_NGX_D3D11_CreateFeature"));
    auto evaluate = reinterpret_cast<PfnEvaluate>(GetProcAddress(snippet, "NVSDK_NGX_D3D11_EvaluateFeature"));
    auto release = reinterpret_cast<PfnRelease>(GetProcAddress(snippet, "NVSDK_NGX_D3D11_ReleaseFeature"));
    auto populate = reinterpret_cast<PfnPopulate>(GetProcAddress(snippet, "NVSDK_NGX_D3D11_PopulateParameters_Impl"));
    auto b_init = reinterpret_cast<PfnBridgeInit>(GetProcAddress(bridge, "NVNGXBridge_D3D12_InitExt"));
    auto b_populate = reinterpret_cast<PfnBridgePopulate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_PopulateParameters"));
    auto b_create = reinterpret_cast<PfnBridgeCreate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_CreateFeature"));
    auto b_evaluate = reinterpret_cast<PfnBridgeEvaluate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_EvaluateFeature"));
    auto core_init = reinterpret_cast<PfnCoreInit>(GetProcAddress(core, "NVSDK_NGX_D3D11_Init"));
    auto core_params = reinterpret_cast<PfnGetParams>(GetProcAddress(core, "NVSDK_NGX_D3D11_GetCapabilityParameters"));
    std::printf("exports: init=%p create=%p evaluate=%p release=%p populate=%p core_init=%p core_params=%p\n",
        snippet_init, create, evaluate, release, populate, core_init, core_params);
    if (!snippet_init || !create || !evaluate || !release || !populate || !b_init || !b_populate || !b_create ||
        !b_evaluate || !core_init || !core_params) return 1;

    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    std::wstring data = std::wstring(temp) + L"nr11";
    CreateDirectoryW(data.c_str(), nullptr);
    NVSDK_NGX_Result r = core_init(kGenericCustomCoreId, data.c_str(), dev, NVSDK_NGX_Version_API);
    std::printf("core D3D11_Init = 0x%08X\n", static_cast<unsigned>(r));
    if (NVSDK_NGX_FAILED(r)) return 1;
    r = b_init(snippet_init, kGenericCustomCoreId, snippet_path.c_str(), dev, NVSDK_NGX_Version_API, nullptr);
    std::printf("snippet D3D11_Init_Ext = 0x%08X\n", static_cast<unsigned>(r));
    if (NVSDK_NGX_FAILED(r)) return 1;
    NVSDK_NGX_Parameter *p = nullptr;
    r = core_params(&p);
    std::printf("GetCapabilityParameters = 0x%08X params=%p\n", static_cast<unsigned>(r), p);
    if (NVSDK_NGX_FAILED(r) || !p) return 1;

    std::vector<unsigned char> pat = Pattern(w, h);
    std::vector<float> zeros(static_cast<size_t>(w) * h * 2, 0.0f);
    const UINT rw = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
    ID3D11Texture2D *color = MakeTexture(dev, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, rw, pat.data(), w * 4);
    ID3D11Texture2D *output = MakeTexture(dev, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, rw, nullptr, 0);
    ID3D11Texture2D *depth = MakeTexture(dev, w, h, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE, zeros.data(), w * 4);
    ID3D11Texture2D *mv = MakeTexture(dev, w, h, DXGI_FORMAT_R32G32_FLOAT, D3D11_BIND_SHADER_RESOURCE, zeros.data(), w * 8);
    if (!color || !output || !depth || !mv) return 1;

    p->Reset();
    b_populate(populate, p);
    p->Set("CreationNodeMask", 1u);
    p->Set("VisibilityNodeMask", 1u);
    p->Set("ResourceWidth", w); p->Set("ResourceHeight", h);
    p->Set("ResourceOutWidth", w); p->Set("ResourceOutHeight", h);
    p->Set("Output.Width", w); p->Set("Output.Height", h);
    p->Set("PerfQualityValue", static_cast<int>(NVSDK_NGX_PerfQuality_Value_UltraQuality));
    p->Set("DLSS.Feature.Create.Flags", static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
        NVSDK_NGX_DLSS_Feature_Flags_AutoExposure | NVSDK_NGX_DLSS_Feature_Flags_DepthInverted));
    p->Set("DLSS.Enable.Output.Subrects", 0);
    p->Set("DLSS.Denoise.Mode", 1);
    p->Set("DLSS.Roughness.Mode", 0u);
    p->Set("DLSS.Use.HW.Depth", 1u);
    SetSizes(p, w, h);
    SetControls(p);
    NVSDK_NGX_Handle *handle = nullptr;
    DWORD code = 0;
    r = SehCreate(b_create, create, ctx, p, &handle, &code);
    std::printf("D3D11_CreateFeature(18) %ux%u = 0x%08X seh=0x%08lX handle=%p\n", w, h, static_cast<unsigned>(r), code, handle);
    if (NVSDK_NGX_FAILED(r) || !handle) return 2;

    LARGE_INTEGER f0, f1, freq;
    QueryPerformanceFrequency(&freq);
    for (int f = 0; f < frames; ++f) {
        p->Set("Color", static_cast<ID3D11Resource *>(color));
        p->Set("Output", static_cast<ID3D11Resource *>(output));
        p->Set("Depth", static_cast<ID3D11Resource *>(depth));
        p->Set("MotionVectors", static_cast<ID3D11Resource *>(mv));
        p->Set("DLSSNR.Color", static_cast<ID3D11Resource *>(color));
        p->Set("DLSSNR.Output", static_cast<ID3D11Resource *>(output));
        p->Set("DLSSNR.Depth", static_cast<ID3D11Resource *>(depth));
        p->Set("DLSSNR.MVec", static_cast<ID3D11Resource *>(mv));
        p->Set("Reset", f == 0 ? 1 : 0);
        p->Set("DLSSNR.Reset", f == 0 ? 1 : 0);
        p->Set("Jitter.Offset.X", 0.0f); p->Set("Jitter.Offset.Y", 0.0f);
        p->Set("MV.Scale.X", 1.0f); p->Set("MV.Scale.Y", 1.0f);
        p->Set("DLSSNR.JitterOffsetX", 0.0f); p->Set("DLSSNR.JitterOffsetY", 0.0f);
        p->Set("DLSSNR.MVecScaleX", 1.0f); p->Set("DLSSNR.MVecScaleY", 1.0f);
        p->Set("DLSS.Pre.Exposure", 1.0f); p->Set("DLSS.Exposure.Scale", 1.0f);
        p->Set("DLSS.Render.Subrect.Dimensions.Width", w);
        p->Set("DLSS.Render.Subrect.Dimensions.Height", h);
        for (const char *name : {"Color", "MVec", "Depth", "Output"}) {
            char key[64];
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseX", name); p->Set(key, 0);
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseY", name); p->Set(key, 0);
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectWidth", name); p->Set(key, static_cast<int>(w));
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectHeight", name); p->Set(key, static_cast<int>(h));
        }
        p->Set("DLSSNR.DepthInverted", 1u);
        SetSizes(p, w, h);
        SetControls(p);
        QueryPerformanceCounter(&f0);
        code = 0;
        r = SehEvaluate(b_evaluate, evaluate, ctx, handle, p, &code);
        ctx->Flush();
        QueryPerformanceCounter(&f1);
        std::printf("frame %d: D3D11_EvaluateFeature = 0x%08X seh=0x%08lX  %.1f ms (CPU side)\n", f, static_cast<unsigned>(r),
            code, (f1.QuadPart - f0.QuadPart) * 1000.0 / freq.QuadPart);
        if (NVSDK_NGX_FAILED(r)) return 3;
    }

    D3D11_TEXTURE2D_DESC sd = {};
    output->GetDesc(&sd);
    sd.Usage = D3D11_USAGE_STAGING; sd.BindFlags = 0; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D *staging = nullptr;
    dev->CreateTexture2D(&sd, nullptr, &staging);
    ctx->CopyResource(staging, output);
    D3D11_MAPPED_SUBRESOURCE m = {};
    hr = ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m);
    if (FAILED(hr)) { std::printf("Map failed 0x%08lX\n", hr); return 4; }
    FILE *fp = nullptr;
    _wfopen_s(&fp, out.c_str(), L"wb");
    std::fprintf(fp, "P6\n%u %u\n255\n", w, h);
    unsigned long long sum = 0, changed = 0;
    for (UINT y = 0; y < h; ++y) {
        const unsigned char *row = static_cast<const unsigned char *>(m.pData) + static_cast<size_t>(y) * m.RowPitch;
        for (UINT x = 0; x < w; ++x) {
            std::fwrite(row + x * 4, 1, 3, fp);
            for (int c = 0; c < 3; ++c) {
                sum += row[x * 4 + c];
                changed += row[x * 4 + c] != pat[(static_cast<size_t>(y) * w + x) * 4 + c];
            }
        }
    }
    std::fclose(fp);
    ctx->Unmap(staging, 0);
    std::printf("output mean %.2f, %.1f%% of values differ from the input\n", sum / (w * h * 3.0), changed * 100.0 / (w * h * 3.0));
    std::printf("VERDICT %s\n", sum == 0 ? "output is black" : (changed == 0 ? "output equals input" : "D3D11 path produced an image"));
    release(handle);
    return 0;
}
