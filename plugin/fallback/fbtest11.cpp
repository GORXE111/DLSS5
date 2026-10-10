// Minimal D3D11 "game" for testing the fallback proxy's D3D11 path: shows a PPM image every frame.
//   fbtest11.exe image.ppm [frames] [options]
//     --model flip|blt        swap effect: FLIP_DISCARD with 2 buffers (default) or the old SEQUENTIAL blt model
//     --legacy                create device and swap chain with D3D11CreateDeviceAndSwapChain (old games; implies blt)
//     --format rgba8|bgra8|rgba8srgb|rgb10    back buffer format (default rgba8; rgba8srgb needs the blt model)
//     --resize N              at frame N resize the window and buffers to 3/4 size
//     --recreate N            at frame N destroy the swap chain and create a new one on the same window
//     --pan N                 shift the image N pixels right every frame (wrapping)
//     --readback file.ppm     after the last Present read the back buffer and save it (blt model only: the buffer
//                             keeps its contents, so this is what the proxy left in it)
// Put the proxy dxgi.dll (and its files) next to fbtest11.exe; it links dxgi.dll by name, so the proxy loads first.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool ReadPpm(const char *path, UINT *w, UINT *h, std::vector<uint8_t> *rgb)
{
    FILE *f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    int max = 0;
    if (std::fscanf(f, "P6 %u %u %d", w, h, &max) != 3 || max != 255) { std::fclose(f); return false; }
    std::fgetc(f);
    rgb->resize(static_cast<size_t>(*w) * *h * 3);
    size_t got = std::fread(rgb->data(), 1, rgb->size(), f);
    std::fclose(f);
    return got == rgb->size();
}

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::printf("FAIL %s = 0x%08X (line %d)\n", #x, (unsigned)hr_, __LINE__); return 1; } } while (0)

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) { return DefWindowProcW(h, m, w, l); }

// image -> one back buffer pixel row layout (w x h, top-left corner of the image, shifted right by `shift`)
static std::vector<uint8_t> Pack(const std::vector<uint8_t> &rgb, UINT W, UINT H, UINT w, UINT h, DXGI_FORMAT fmt, int shift)
{
    std::vector<uint8_t> out(static_cast<size_t>(w) * h * 4, 128);
    for (UINT y = 0; y < std::min(h, H); ++y) {
        for (UINT x = 0; x < std::min(w, W); ++x) {
            const uint8_t *s = &rgb[(static_cast<size_t>(y) * W + ((x + W - shift % W) % W)) * 3];
            uint8_t *d = &out[(static_cast<size_t>(y) * w + x) * 4];
            if (fmt == DXGI_FORMAT_R10G10B10A2_UNORM) {
                auto ten = [](uint8_t v) { return static_cast<uint32_t>(v * 1023.0f / 255.0f + 0.5f); };
                uint32_t v = ten(s[0]) | (ten(s[1]) << 10) | (ten(s[2]) << 20) | (3u << 30);
                std::memcpy(d, &v, 4);
            } else if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM) {
                d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; d[3] = 255;
            } else {
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
            }
        }
    }
    return out;
}

int main(int argc, char **argv)
{
    if (argc < 2) { std::printf("usage: fbtest11 image.ppm [frames] [--model flip|blt] [--legacy] [--format ...] ...\n"); return 2; }
    UINT W, H;
    std::vector<uint8_t> rgb;
    if (!ReadPpm(argv[1], &W, &H, &rgb)) { std::printf("cannot read %s\n", argv[1]); return 2; }
    int frames = 60, resize_at = -1, recreate_at = -1, pan = 0;
    bool flip = true, legacy = false;
    const char *readback = nullptr;
    DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? std::atoi(argv[++i]) : 0; };
        if (a == "--model" && i + 1 < argc) flip = std::string(argv[++i]) == "flip";
        else if (a == "--legacy") { legacy = true; flip = false; }
        else if (a == "--format" && i + 1 < argc) {
            std::string f = argv[++i];
            fmt = f == "bgra8" ? DXGI_FORMAT_B8G8R8A8_UNORM : f == "rgba8srgb" ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB :
                f == "rgb10" ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
        }
        else if (a == "--resize") resize_at = next();
        else if (a == "--recreate") recreate_at = next();
        else if (a == "--pan") pan = next();
        else if (a == "--readback" && i + 1 < argc) readback = argv[++i];
        else if (a[0] != '-') frames = std::atoi(a.c_str());
        else { std::printf("unknown option %s\n", a.c_str()); return 2; }
    }

    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"dlss5fbtest11";
    RegisterClassW(&wc);
    RECT r = {0, 0, LONG(W), LONG(H)};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"dlss5 fallback test (D3D11)", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 40, 40,
        r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);

    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *ctx = nullptr;
    IDXGISwapChain *sc = nullptr;
    IDXGIFactory2 *factory = nullptr;
    UINT w = W, h = H;
    auto make_chain = [&]() -> HRESULT {
        if (legacy) {
            DXGI_SWAP_CHAIN_DESC sd = {};
            sd.BufferDesc.Width = w; sd.BufferDesc.Height = h; sd.BufferDesc.Format = fmt;
            sd.SampleDesc.Count = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.BufferCount = 1;
            sd.OutputWindow = hwnd; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_SEQUENTIAL;
            if (device == nullptr)
                return D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                    D3D11_SDK_VERSION, &sd, &sc, &device, nullptr, &ctx);
            IDXGIDevice *dd = nullptr; IDXGIAdapter *ad = nullptr; IDXGIFactory *f = nullptr;
            device->QueryInterface(IID_PPV_ARGS(&dd)); dd->GetAdapter(&ad); ad->GetParent(IID_PPV_ARGS(&f));
            HRESULT hr = f->CreateSwapChain(device, &sd, &sc);
            f->Release(); ad->Release(); dd->Release();
            return hr;
        }
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = w; sd.Height = h; sd.Format = fmt; sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = flip ? 2 : 1;
        sd.SwapEffect = flip ? DXGI_SWAP_EFFECT_FLIP_DISCARD : DXGI_SWAP_EFFECT_SEQUENTIAL;
        IDXGISwapChain1 *sc1 = nullptr;
        HRESULT hr = factory->CreateSwapChainForHwnd(device, hwnd, &sd, nullptr, nullptr, &sc1);
        sc = sc1;
        return hr;
    };
    if (!legacy) {
        CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
        CHECK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &ctx));
    }
    CHECK(make_chain());

    LARGE_INTEGER t0, t1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    for (int f = 0; f < frames; ++f) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        if (f == recreate_at) {
            sc->Release();
            sc = nullptr;
            ctx->ClearState();
            ctx->Flush();
            CHECK(make_chain());
        }
        if (f == resize_at) {
            w = W * 3 / 4; h = H * 3 / 4;
            RECT rr = {0, 0, LONG(w), LONG(h)};
            AdjustWindowRect(&rr, WS_OVERLAPPEDWINDOW, FALSE);
            SetWindowPos(hwnd, nullptr, 0, 0, rr.right - rr.left, rr.bottom - rr.top, SWP_NOMOVE | SWP_NOZORDER);
            CHECK(sc->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0));
        }
        ID3D11Texture2D *bb = nullptr;
        CHECK(sc->GetBuffer(0, IID_PPV_ARGS(&bb)));
        std::vector<uint8_t> px = Pack(rgb, W, H, w, h, fmt, pan * f);
        ctx->UpdateSubresource(bb, 0, nullptr, px.data(), w * 4, 0);
        bb->Release();
        CHECK(sc->Present(0, 0));
    }
    QueryPerformanceCounter(&t1);

    if (readback) {
        ID3D11Texture2D *bb = nullptr, *staging = nullptr;
        CHECK(sc->GetBuffer(0, IID_PPV_ARGS(&bb)));
        D3D11_TEXTURE2D_DESC d = {};
        bb->GetDesc(&d);
        d.Usage = D3D11_USAGE_STAGING; d.BindFlags = 0; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0;
        CHECK(device->CreateTexture2D(&d, nullptr, &staging));
        ctx->CopyResource(staging, bb);
        D3D11_MAPPED_SUBRESOURCE m = {};
        CHECK(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m));
        FILE *fp = std::fopen(readback, "wb");
        std::fprintf(fp, "P6\n%u %u\n255\n", d.Width, d.Height);
        const bool bgr = d.Format == DXGI_FORMAT_B8G8R8A8_UNORM;
        for (UINT y = 0; y < d.Height; ++y) {
            const uint8_t *row = static_cast<const uint8_t *>(m.pData) + static_cast<size_t>(y) * m.RowPitch;
            for (UINT x = 0; x < d.Width; ++x) {
                uint8_t p[3] = {row[x * 4 + (bgr ? 2 : 0)], row[x * 4 + 1], row[x * 4 + (bgr ? 0 : 2)]};
                std::fwrite(p, 1, 3, fp);
            }
        }
        std::fclose(fp);
        ctx->Unmap(staging, 0);
        staging->Release();
        bb->Release();
    }
    sc->Release();
    ctx->ClearState();
    ctx->Flush();
    ctx->Release();
    device->Release();
    if (factory) factory->Release();
    std::printf("OK %d frames, %.2f ms/frame\n", frames, (t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart / frames);
    return 0;
}
