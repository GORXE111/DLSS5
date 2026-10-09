// Minimal D3D12 "game" for testing the fallback proxy: shows a PPM image every frame.
//   fbtest.exe image.ppm [frames] [options]
//     --format rgba8|bgra8|rgb10|rgba16f   back buffer format (default rgba8)
//     --present1                           present with IDXGISwapChain1::Present1
//     --resize N                           at frame N resize the window and buffers to 3/4 size (ResizeBuffers)
//     --resize1 N                          same with ResizeBuffers1
//     --recreate N                         at frame N destroy the swap chain and create a new one on the same window
//     --waitable                           FRAME_LATENCY_WAITABLE_OBJECT swap chain, wait on it every frame
//     --chains 2                           two windows / swap chains presented alternately
//     --fullscreen N                       at frame N enter fullscreen, leave again 30 frames later
//     --pan N                              shift the image N pixels right every frame (wrapping, like nr-lab --temporal-shift)
//     --cut N image2.ppm                   from frame N on show image2 (same size): a hard scene cut
// The image is copied into the top-left corner of each back buffer (the rest is cleared to grey).
// Put the proxy dxgi.dll (and its files) next to fbtest.exe; it links dxgi.dll by name, so the proxy loads first.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cmath>
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

static uint16_t Half(float v)
{
    // exact for the values used here (k/255 in [0,1])
    uint32_t x;
    std::memcpy(&x, &v, 4);
    if (v == 0.0f) return 0;
    int e = static_cast<int>((x >> 23) & 255) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return 0;   // not needed for k/255
    uint32_t r = (static_cast<uint32_t>(e) << 10) | (m >> 13);
    if ((m >> 12) & 1) r += 1;   // round half up
    return static_cast<uint16_t>(r);
}

// RGB8 pixels -> one row-major texel array in the back buffer format
static std::vector<uint8_t> Encode(const std::vector<uint8_t> &rgb, size_t n, DXGI_FORMAT fmt, UINT *bpp)
{
    std::vector<uint8_t> out;
    if (fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
        *bpp = 8;
        out.resize(n * 8);
        for (size_t i = 0; i < n; ++i) {
            uint16_t px[4] = {Half(rgb[3 * i] / 255.0f), Half(rgb[3 * i + 1] / 255.0f), Half(rgb[3 * i + 2] / 255.0f), Half(1.0f)};
            std::memcpy(&out[8 * i], px, 8);
        }
        return out;
    }
    *bpp = 4;
    out.resize(n * 4);
    for (size_t i = 0; i < n; ++i) {
        uint8_t r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
        if (fmt == DXGI_FORMAT_R10G10B10A2_UNORM) {
            auto q = [](uint8_t v) { return static_cast<uint32_t>(std::lround(v / 255.0 * 1023.0)); };
            uint32_t v = q(r) | (q(g) << 10) | (q(b) << 20) | (3u << 30);
            std::memcpy(&out[4 * i], &v, 4);
        } else if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM) {
            out[4 * i] = b; out[4 * i + 1] = g; out[4 * i + 2] = r; out[4 * i + 3] = 255;
        } else {
            out[4 * i] = r; out[4 * i + 1] = g; out[4 * i + 2] = b; out[4 * i + 3] = 255;
        }
    }
    return out;
}

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::printf("FAIL %s = 0x%08X (line %d)\n", #x, (unsigned)hr_, __LINE__); return 1; } } while (0)

struct Chain {
    HWND hwnd = nullptr;
    IDXGISwapChain3 *sc = nullptr;
    UINT w = 0, h = 0;
    HANDLE waitable = nullptr;
};

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) { return DefWindowProcW(h, m, w, l); }

int main(int argc, char **argv)
{
    if (argc < 2) { std::printf("usage: fbtest image.ppm [frames] [--format ...] [--present1] [--resize N] ...\n"); return 2; }
    UINT W, H;
    std::vector<uint8_t> rgb;
    if (!ReadPpm(argv[1], &W, &H, &rgb)) { std::printf("cannot read %s\n", argv[1]); return 2; }
    int frames = 60, resize_at = -1, recreate_at = -1, fullscreen_at = -1, chains = 1, pan = 0, cut_at = -1;
    const char *cut_image = nullptr;
    bool present1 = false, resize1 = false, waitable = false;
    DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? std::atoi(argv[++i]) : 0; };
        if (a == "--format" && i + 1 < argc) {
            std::string f = argv[++i];
            fmt = f == "bgra8" ? DXGI_FORMAT_B8G8R8A8_UNORM : f == "rgb10" ? DXGI_FORMAT_R10G10B10A2_UNORM
                : f == "rgba16f" ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
        }
        else if (a == "--present1") present1 = true;
        else if (a == "--resize") resize_at = next();
        else if (a == "--resize1") { resize_at = next(); resize1 = true; }
        else if (a == "--recreate") recreate_at = next();
        else if (a == "--waitable") waitable = true;
        else if (a == "--chains") chains = std::max(1, std::min(2, next()));
        else if (a == "--fullscreen") fullscreen_at = next();
        else if (a == "--pan") pan = next();
        else if (a == "--cut" && i + 2 < argc) { cut_at = std::atoi(argv[++i]); cut_image = argv[++i]; }
        else if (a[0] != '-') frames = std::atoi(a.c_str());
        else { std::printf("unknown option %s\n", a.c_str()); return 2; }
    }

    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"dlss5fbtest";
    RegisterClassW(&wc);

    IDXGIFactory4 *factory = nullptr;
    CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ID3D12Device *device = nullptr;
    CHECK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT};
    ID3D12CommandQueue *queue = nullptr;
    CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    const UINT sc_flags = waitable ? DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT : 0;

    auto make_chain = [&](Chain &c, int index) -> HRESULT {
        if (c.hwnd == nullptr) {
            RECT r = {0, 0, LONG(W), LONG(H)};
            AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
            c.hwnd = CreateWindowW(wc.lpszClassName, index ? L"dlss5 fallback test 2" : L"dlss5 fallback test",
                WS_OVERLAPPEDWINDOW | WS_VISIBLE, 40 + 60 * index, 40 + 60 * index, r.right - r.left, r.bottom - r.top,
                nullptr, nullptr, wc.hInstance, nullptr);
        }
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = W;
        sd.Height = H;
        sd.Format = fmt;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 3;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.Flags = sc_flags;
        IDXGISwapChain1 *sc1 = nullptr;
        HRESULT hr = factory->CreateSwapChainForHwnd(queue, c.hwnd, &sd, nullptr, nullptr, &sc1);
        if (FAILED(hr)) return hr;
        hr = sc1->QueryInterface(IID_PPV_ARGS(&c.sc));
        sc1->Release();
        if (FAILED(hr)) return hr;
        c.w = W;
        c.h = H;
        if (waitable) {
            c.sc->SetMaximumFrameLatency(1);
            c.waitable = c.sc->GetFrameLatencyWaitableObject();
        }
        return S_OK;
    };
    std::vector<Chain> sc(chains);
    for (int i = 0; i < chains; ++i) CHECK(make_chain(sc[i], i));

    // the image, once, in a default texture of the back buffer format
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = W;
    td.Height = H * (cut_image ? 2 : 1);
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = fmt;
    td.SampleDesc.Count = 1;
    D3D12_HEAP_PROPERTIES dh = {D3D12_HEAP_TYPE_DEFAULT}, uh = {D3D12_HEAP_TYPE_UPLOAD};
    ID3D12Resource *tex = nullptr, *upload = nullptr;
    CHECK(device->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex)));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT64 total = 0;
    device->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total;
    bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CHECK(device->CreateCommittedResource(&uh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)));
    UINT bpp = 4;
    if (cut_image) {
        // the cut image goes into the lower half of a double-height texture: one upload, two copy sources
        UINT W2, H2;
        std::vector<uint8_t> rgb2;
        if (!ReadPpm(cut_image, &W2, &H2, &rgb2) || W2 != W || H2 != H) { std::printf("cannot use %s\n", cut_image); return 2; }
        rgb.insert(rgb.end(), rgb2.begin(), rgb2.end());
    }
    std::vector<uint8_t> texels = Encode(rgb, static_cast<size_t>(W) * H * (cut_image ? 2 : 1), fmt, &bpp);
    uint8_t *p = nullptr;
    upload->Map(0, nullptr, reinterpret_cast<void **>(&p));
    for (UINT y = 0; y < td.Height; ++y) std::memcpy(p + fp.Offset + y * fp.Footprint.RowPitch, &texels[static_cast<size_t>(y) * W * bpp], W * bpp);
    upload->Unmap(0, nullptr);

    ID3D12CommandAllocator *alloc = nullptr;
    ID3D12GraphicsCommandList *list = nullptr;
    ID3D12Fence *fence = nullptr;
    ID3D12DescriptorHeap *rtv_heap = nullptr;
    CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
    CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&list)));
    CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    D3D12_DESCRIPTOR_HEAP_DESC rd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK(device->CreateDescriptorHeap(&rd, IID_PPV_ARGS(&rtv_heap)));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;
    auto wait = [&] {
        queue->Signal(fence, ++fv);
        fence->SetEventOnCompletion(fv, event);
        WaitForSingleObject(event, INFINITE);
    };
    D3D12_TEXTURE_COPY_LOCATION dst = {tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION src = {upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint = fp;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER b = {};
    b.Transition = {tex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &b);
    list->Close();
    ID3D12CommandList *lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    wait();

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    for (int i = 0; i < frames; ++i) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&msg);

        if (i == resize_at) {
            for (Chain &c : sc) {
                c.w = std::max(64u, W * 3 / 4);
                c.h = std::max(64u, H * 3 / 4);
                RECT r = {0, 0, LONG(c.w), LONG(c.h)};
                AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
                SetWindowPos(c.hwnd, nullptr, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER);
                if (resize1) {
                    UINT masks[3] = {1, 1, 1};
                    IUnknown *queues[3] = {queue, queue, queue};
                    CHECK(c.sc->ResizeBuffers1(3, c.w, c.h, DXGI_FORMAT_UNKNOWN, sc_flags, masks, queues));
                } else {
                    CHECK(c.sc->ResizeBuffers(0, c.w, c.h, DXGI_FORMAT_UNKNOWN, sc_flags));
                }
            }
            std::printf("frame %d: resized to %ux%u%s\n", i, sc[0].w, sc[0].h, resize1 ? " (ResizeBuffers1)" : "");
        }
        if (i == recreate_at) {
            for (int k = 0; k < chains; ++k) {
                if (sc[k].waitable) CloseHandle(sc[k].waitable);
                ULONG left = sc[k].sc->Release();
                sc[k].sc = nullptr;
                if (left != 0) std::printf("WARN swap chain still has %lu references\n", left);
                CHECK(make_chain(sc[k], k));
            }
            std::printf("frame %d: swap chains recreated\n", i);
        }
        if (fullscreen_at >= 0 && (i == fullscreen_at || i == fullscreen_at + 30)) {
            BOOL on = i == fullscreen_at;
            HRESULT hr = sc[0].sc->SetFullscreenState(on, nullptr);
            std::printf("frame %d: SetFullscreenState(%d) = 0x%08X\n", i, on, static_cast<unsigned>(hr));
            DXGI_SWAP_CHAIN_DESC1 d1 = {};
            sc[0].sc->GetDesc1(&d1);
            CHECK(sc[0].sc->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, sc_flags));
            sc[0].sc->GetDesc1(&d1);
            sc[0].w = d1.Width;
            sc[0].h = d1.Height;
        }

        for (Chain &c : sc) {
            if (c.waitable) WaitForSingleObjectEx(c.waitable, 1000, TRUE);
            ID3D12Resource *bb = nullptr;
            CHECK(c.sc->GetBuffer(c.sc->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&bb)));
            alloc->Reset();
            list->Reset(alloc, nullptr);
            b.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET};
            list->ResourceBarrier(1, &b);
            D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            device->CreateRenderTargetView(bb, nullptr, rtv);
            const float grey[4] = {0.2f, 0.2f, 0.2f, 1.0f};
            list->ClearRenderTargetView(rtv, grey, 0, nullptr);
            b.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST};
            list->ResourceBarrier(1, &b);
            D3D12_TEXTURE_COPY_LOCATION to = {bb, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION from = {tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            const UINT y0 = cut_image && i >= cut_at ? H : 0;   // source rows of the image shown this frame
            if (pan != 0 && c.w == W && c.h == H) {
                // frame i shows the image moved right by i*pan pixels, wrapping around
                const UINT off = static_cast<UINT>((static_cast<long long>(i) * pan % W + W) % W);
                D3D12_BOX right = {0, y0, 0, W - off, y0 + H, 1}, left = {W - off, y0, 0, W, y0 + H, 1};
                list->CopyTextureRegion(&to, off, 0, 0, &from, &right);
                if (off) list->CopyTextureRegion(&to, 0, 0, 0, &from, &left);
            } else {
                D3D12_BOX box = {0, y0, 0, std::min(W, c.w), y0 + std::min(H, c.h), 1};
                list->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
            }
            b.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT};
            list->ResourceBarrier(1, &b);
            list->Close();
            queue->ExecuteCommandLists(1, lists);
            bb->Release();
            if (present1) {
                DXGI_PRESENT_PARAMETERS pp = {};
                CHECK(c.sc->Present1(0, 0, &pp));
            } else {
                CHECK(c.sc->Present(0, 0));
            }
            wait();
        }
    }
    QueryPerformanceCounter(&t1);
    std::printf("%d frames, %.2f ms/frame\n", frames, (t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart / frames);
    for (Chain &c : sc) {
        c.sc->SetFullscreenState(FALSE, nullptr);
        if (c.waitable) CloseHandle(c.waitable);
        c.sc->Release();
        DestroyWindow(c.hwnd);
    }
    wait();
    std::printf("OK\n");
    return 0;
}
