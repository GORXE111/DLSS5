// Minimal D3D12 "game" for testing the fallback proxy: shows a PPM image every frame.
//   fbtest.exe image.ppm [frames]
// Put the proxy dxgi.dll (and its files) next to fbtest.exe; it links dxgi.dll by name, so the proxy loads first.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static bool ReadPpm(const char *path, UINT *w, UINT *h, std::vector<uint8_t> *rgba)
{
    FILE *f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    int max = 0;
    if (std::fscanf(f, "P6 %u %u %d", w, h, &max) != 3 || max != 255) { std::fclose(f); return false; }
    std::fgetc(f);
    std::vector<uint8_t> rgb(static_cast<size_t>(*w) * *h * 3);
    size_t got = std::fread(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
    if (got != rgb.size()) return false;
    rgba->resize(static_cast<size_t>(*w) * *h * 4);
    for (size_t i = 0; i < static_cast<size_t>(*w) * *h; ++i) {
        (*rgba)[4 * i] = rgb[3 * i];
        (*rgba)[4 * i + 1] = rgb[3 * i + 1];
        (*rgba)[4 * i + 2] = rgb[3 * i + 2];
        (*rgba)[4 * i + 3] = 255;
    }
    return true;
}

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::printf("FAIL %s = 0x%08X\n", #x, (unsigned)hr_); return 1; } } while (0)

int main(int argc, char **argv)
{
    if (argc < 2) { std::printf("usage: fbtest image.ppm [frames]\n"); return 2; }
    UINT W, H;
    std::vector<uint8_t> image;
    if (!ReadPpm(argv[1], &W, &H, &image)) { std::printf("cannot read %s\n", argv[1]); return 2; }
    const int frames = argc > 2 ? std::atoi(argv[2]) : 60;

    WNDCLASSW wc = {};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"dlss5fbtest";
    RegisterClassW(&wc);
    RECT r = {0, 0, LONG(W), LONG(H)};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"dlss5 fallback test", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);

    IDXGIFactory4 *factory = nullptr;
    CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ID3D12Device *device = nullptr;
    CHECK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT};
    ID3D12CommandQueue *queue = nullptr;
    CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = W;
    sd.Height = H;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    IDXGISwapChain1 *sc1 = nullptr;
    CHECK(factory->CreateSwapChainForHwnd(queue, hwnd, &sd, nullptr, nullptr, &sc1));
    IDXGISwapChain3 *sc = nullptr;
    CHECK(sc1->QueryInterface(IID_PPV_ARGS(&sc)));

    // upload the image once into a default texture
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = W;
    td.Height = H;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
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
    uint8_t *p = nullptr;
    upload->Map(0, nullptr, reinterpret_cast<void **>(&p));
    for (UINT y = 0; y < H; ++y) memcpy(p + fp.Offset + y * fp.Footprint.RowPitch, &image[static_cast<size_t>(y) * W * 4], W * 4);
    upload->Unmap(0, nullptr);

    ID3D12CommandAllocator *alloc = nullptr;
    ID3D12GraphicsCommandList *list = nullptr;
    ID3D12Fence *fence = nullptr;
    CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
    CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&list)));
    CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
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
        ID3D12Resource *bb = nullptr;
        CHECK(sc->GetBuffer(sc->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&bb)));
        alloc->Reset();
        list->Reset(alloc, nullptr);
        b.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST};
        list->ResourceBarrier(1, &b);
        list->CopyResource(bb, tex);
        b.Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT};
        list->ResourceBarrier(1, &b);
        list->Close();
        queue->ExecuteCommandLists(1, lists);
        bb->Release();
        CHECK(sc->Present(0, 0));
        wait();
    }
    QueryPerformanceCounter(&t1);
    std::printf("%d frames, %.2f ms/frame\n", frames, (t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart / frames);
    DestroyWindow(hwnd);
    return 0;
}
