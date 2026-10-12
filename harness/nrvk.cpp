// nrvk: does DLSS-NR work through its Vulkan entry points?
// Creates a Vulkan device, initialises NGX (driver core + nvngx_dlssnr.dll through the caller-identity bridge),
// runs N frames on the same synthetic pattern nr-lab uses and writes the last output as a PPM
// (frame 0 of nr-lab on the same pattern is the reference to compare with).
//
//   nrvk.exe [--dir <folder with nvngx_dlssnr.dll and nvngx.dll>] [--size WxH] [--frames N] [--out file.ppm]
//
// Needs the Vulkan headers (github.com/KhronosGroup/Vulkan-Headers in ..\oss\Vulkan-Headers, not part of this
// repository); vulkan-1.dll is loaded at run time, no Vulkan SDK required. Build: see build.bat.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VK_NO_PROTOTYPES
#include <windows.h>
#include <vulkan/vulkan.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs_vk.h>
#include <nvsdk_ngx_params.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr NVSDK_NGX_Feature kFeatureDlssNr = static_cast<NVSDK_NGX_Feature>(0x12);
static constexpr unsigned long long kGenericCustomCoreId = 0x0876232CULL;

using PfnCoreInit = NVSDK_NGX_Result (NVSDK_CONV *)(unsigned long long, const wchar_t *, VkInstance, VkPhysicalDevice,
    VkDevice, NVSDK_NGX_Version);
using PfnSnippetInit = NVSDK_NGX_Result (NVSDK_CONV *)(unsigned long long, const wchar_t *, VkInstance, VkPhysicalDevice,
    VkDevice, NVSDK_NGX_Version, const NVSDK_NGX_Parameter *);
using PfnGetParams = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Parameter **);
using PfnPopulate = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Parameter *);
using PfnCreate = NVSDK_NGX_Result (NVSDK_CONV *)(VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
using PfnEvaluate = NVSDK_NGX_Result (NVSDK_CONV *)(VkCommandBuffer, const NVSDK_NGX_Handle *, const NVSDK_NGX_Parameter *,
    PFN_NVSDK_NGX_ProgressCallback_C);
using PfnRelease = NVSDK_NGX_Result (NVSDK_CONV *)(NVSDK_NGX_Handle *);
using PfnBridgeInit = NVSDK_NGX_Result (NVSDK_CONV *)(PfnSnippetInit, unsigned long long, const wchar_t *, VkInstance,
    VkPhysicalDevice, VkDevice, NVSDK_NGX_Version, const NVSDK_NGX_Parameter *);
// The other bridge entry points only forward pointers, so the D3D12-named ones work for Vulkan handles too.
using PfnBridgePopulate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnPopulate, NVSDK_NGX_Parameter *);
using PfnBridgeCreate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnCreate, VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *,
    NVSDK_NGX_Handle **);
using PfnBridgeEvaluate = NVSDK_NGX_Result (NVSDK_CONV *)(PfnEvaluate, VkCommandBuffer, const NVSDK_NGX_Handle *,
    const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback_C);

#define VK_FUNCS(X) \
    X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) \
    X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetDeviceQueue) X(vkCreateCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) \
    X(vkCreateImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) X(vkBindImageMemory) X(vkCreateImageView) \
    X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkMapMemory) X(vkCmdPipelineBarrier) \
    X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdClearColorImage) X(vkResetCommandBuffer) X(vkDeviceWaitIdle)
#define X(n) static PFN_##n n;
VK_FUNCS(X)
#undef X

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
static std::vector<unsigned char> Pattern(uint32_t w, uint32_t h)
{
    std::vector<unsigned char> p(static_cast<size_t>(w) * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
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

static VkDevice g_dev;
static VkPhysicalDeviceMemoryProperties g_mem;

static uint32_t MemoryType(uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < g_mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g_mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    NVSDK_NGX_Resource_VK ngx = {};
};

static bool MakeImage(Image *img, uint32_t w, uint32_t h, VkFormat format, bool read_write)
{
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {w, h, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkCreateImage(g_dev, &info, nullptr, &img->image) != VK_SUCCESS) return false;
    VkMemoryRequirements req = {};
    vkGetImageMemoryRequirements(g_dev, img->image, &req);
    VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
        MemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(g_dev, &alloc, nullptr, &mem) != VK_SUCCESS || vkBindImageMemory(g_dev, img->image, mem, 0) != VK_SUCCESS)
        return false;
    VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = img->image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g_dev, &view, nullptr, &img->view) != VK_SUCCESS) return false;
    img->ngx.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    img->ngx.Resource.ImageViewInfo = {img->view, img->image, view.subresourceRange, format, w, h};
    img->ngx.ReadWrite = read_write;
    return true;
}

static void Layout(VkCommandBuffer cmd, const Image &img, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static NVSDK_NGX_Result SehCreate(PfnBridgeCreate b, PfnCreate f, VkCommandBuffer cmd, NVSDK_NGX_Parameter *p,
    NVSDK_NGX_Handle **h, DWORD *code)
{
    __try { return b(f, cmd, kFeatureDlssNr, p, h); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return static_cast<NVSDK_NGX_Result>(0x7fffffff); }
}

static NVSDK_NGX_Result SehEvaluate(PfnBridgeEvaluate b, PfnEvaluate f, VkCommandBuffer cmd, NVSDK_NGX_Handle *h,
    NVSDK_NGX_Parameter *p, DWORD *code)
{
    __try { return b(f, cmd, h, p, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return static_cast<NVSDK_NGX_Result>(0x7fffffff); }
}

static void SetSizes(NVSDK_NGX_Parameter *p, uint32_t w, uint32_t h)
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
    std::wstring out = L"nrvk-output.ppm";
    uint32_t w = 640, h = 360;
    int frames = 1;
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--dir" && i + 1 < argc) { dir = argv[++i]; if (dir.back() != L'\\') dir += L'\\'; }
        else if (a == L"--size" && i + 1 < argc) swscanf_s(argv[++i], L"%ux%u", &w, &h);
        else if (a == L"--frames" && i + 1 < argc) frames = _wtoi(argv[++i]);
        else if (a == L"--out" && i + 1 < argc) out = argv[++i];
    }

    HMODULE vk = LoadLibraryW(L"vulkan-1.dll");
    if (!vk) { std::printf("vulkan-1.dll not found\n"); return 1; }
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vk, "vkGetInstanceProcAddr"));
    vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "nrvk";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    VkResult vr = vkCreateInstance(&ici, nullptr, &instance);
    std::printf("vkCreateInstance = %d\n", vr);
    if (vr != VK_SUCCESS) return 1;
#define X(n) if (std::strcmp(#n, "vkCreateInstance") != 0) n = reinterpret_cast<PFN_##n>(gipa(instance, #n)); if (!n) { std::printf("missing %s\n", #n); return 1; }
    VK_FUNCS(X)
#undef X

    uint32_t count = 8;
    VkPhysicalDevice pds[8];
    vkEnumeratePhysicalDevices(instance, &count, pds);
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pds[i], &props);
        std::printf("device %u: %s (vendor 0x%04X, Vulkan %u.%u.%u)\n", i, props.deviceName, props.vendorID,
            VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
        if (pd == VK_NULL_HANDLE && props.vendorID == 0x10DE) pd = pds[i];
    }
    if (pd == VK_NULL_HANDLE) { std::printf("no NVIDIA device\n"); return 1; }

    // what the snippet's Vulkan backend uses (names found in nvngx_dlssnr.dll): CUDA kernels launched from Vulkan
    const char *wanted[] = {"VK_NVX_binary_import", "VK_NVX_image_view_handle", "VK_KHR_push_descriptor"};
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, exts.data());
    std::vector<const char *> enable;
    for (const char *name : wanted) {
        bool has = false;
        for (const auto &e : exts) has = has || std::strcmp(e.extensionName, name) == 0;
        std::printf("extension %s: %s\n", name, has ? "available" : "MISSING");
        if (has) enable.push_back(name);
    }
    uint32_t qcount = 16;
    VkQueueFamilyProperties qprops[16];
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops);
    uint32_t family = 0;
    for (uint32_t i = 0; i < qcount; ++i)
        if ((qprops[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &prio};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(enable.size());
    dci.ppEnabledExtensionNames = enable.data();
    vr = vkCreateDevice(pd, &dci, nullptr, &g_dev);
    std::printf("vkCreateDevice = %d\n", vr);
    if (vr != VK_SUCCESS) return 1;
    vkGetPhysicalDeviceMemoryProperties(pd, &g_mem);
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(g_dev, family, 0, &queue);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(g_dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(g_dev, &cai, &cmd);
    auto begin = [&] {
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &bi);
    };
    auto submit = [&] {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VkResult r1 = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), r2 = vkQueueWaitIdle(queue);
        if (r1 != VK_SUCCESS || r2 != VK_SUCCESS) std::printf("submit = %d, wait = %d\n", r1, r2);
    };

    std::wstring snippet_path = dir + L"nvngx_dlssnr.dll", bridge_path = dir + L"nvngx.dll";
    HMODULE snippet = LoadLibraryExW(snippet_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE bridge = LoadLibraryExW(bridge_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    HMODULE core = LoadDriverCore();
    std::printf("modules: snippet=%p bridge=%p core=%p\n", snippet, bridge, core);
    if (!snippet || !bridge || !core) return 1;
    auto snippet_init = reinterpret_cast<PfnSnippetInit>(GetProcAddress(snippet, "NVSDK_NGX_VULKAN_Init_Ext"));
    auto create = reinterpret_cast<PfnCreate>(GetProcAddress(snippet, "NVSDK_NGX_VULKAN_CreateFeature"));
    auto evaluate = reinterpret_cast<PfnEvaluate>(GetProcAddress(snippet, "NVSDK_NGX_VULKAN_EvaluateFeature"));
    auto release = reinterpret_cast<PfnRelease>(GetProcAddress(snippet, "NVSDK_NGX_VULKAN_ReleaseFeature"));
    auto populate = reinterpret_cast<PfnPopulate>(GetProcAddress(snippet, "NVSDK_NGX_VULKAN_PopulateParameters_Impl"));
    auto b_init = reinterpret_cast<PfnBridgeInit>(GetProcAddress(bridge, "NVNGXBridge_VULKAN_InitExt"));
    auto b_populate = reinterpret_cast<PfnBridgePopulate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_PopulateParameters"));
    auto b_create = reinterpret_cast<PfnBridgeCreate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_CreateFeature"));
    auto b_evaluate = reinterpret_cast<PfnBridgeEvaluate>(GetProcAddress(bridge, "NVNGXBridge_D3D12_EvaluateFeature"));
    auto core_init = reinterpret_cast<PfnCoreInit>(GetProcAddress(core, "NVSDK_NGX_VULKAN_Init"));
    auto core_params = reinterpret_cast<PfnGetParams>(GetProcAddress(core, "NVSDK_NGX_VULKAN_GetCapabilityParameters"));
    if (!snippet_init || !create || !evaluate || !release || !populate || !b_init || !b_populate || !b_create ||
        !b_evaluate || !core_init || !core_params) { std::printf("an export is missing\n"); return 1; }

    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    std::wstring data = std::wstring(temp) + L"nrvk";
    CreateDirectoryW(data.c_str(), nullptr);
    NVSDK_NGX_Result r = core_init(kGenericCustomCoreId, data.c_str(), instance, pd, g_dev, NVSDK_NGX_Version_API);
    std::printf("core VULKAN_Init = 0x%08X\n", static_cast<unsigned>(r));
    if (NVSDK_NGX_FAILED(r)) return 1;
    r = b_init(snippet_init, kGenericCustomCoreId, snippet_path.c_str(), instance, pd, g_dev, NVSDK_NGX_Version_API, nullptr);
    std::printf("snippet VULKAN_Init_Ext = 0x%08X\n", static_cast<unsigned>(r));
    if (NVSDK_NGX_FAILED(r)) return 1;
    NVSDK_NGX_Parameter *p = nullptr;
    r = core_params(&p);
    std::printf("GetCapabilityParameters = 0x%08X params=%p\n", static_cast<unsigned>(r), p);
    if (NVSDK_NGX_FAILED(r) || !p) return 1;

    Image color, output, depth, mv;
    if (!MakeImage(&color, w, h, VK_FORMAT_R8G8B8A8_UNORM, false) || !MakeImage(&output, w, h, VK_FORMAT_R8G8B8A8_UNORM, true) ||
        !MakeImage(&depth, w, h, VK_FORMAT_R32_SFLOAT, false) || !MakeImage(&mv, w, h, VK_FORMAT_R32G32_SFLOAT, false)) {
        std::printf("image creation failed\n");
        return 1;
    }
    // one host-visible buffer: pattern upload, later the read-back
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    vkCreateBuffer(g_dev, &bci, nullptr, &buffer);
    VkMemoryRequirements breq = {};
    vkGetBufferMemoryRequirements(g_dev, buffer, &breq);
    VkMemoryAllocateInfo balloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, breq.size,
        MemoryType(breq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory bmem = VK_NULL_HANDLE;
    vkAllocateMemory(g_dev, &balloc, nullptr, &bmem);
    vkBindBufferMemory(g_dev, buffer, bmem, 0);
    unsigned char *mapped = nullptr;
    vkMapMemory(g_dev, bmem, 0, bytes, 0, reinterpret_cast<void **>(&mapped));
    std::vector<unsigned char> pat = Pattern(w, h);
    std::memcpy(mapped, pat.data(), pat.size());

    const VkImageLayout kGeneral = VK_IMAGE_LAYOUT_GENERAL, kDst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkBufferImageCopy region = {};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {w, h, 1};
    begin();
    Layout(cmd, color, VK_IMAGE_LAYOUT_UNDEFINED, kDst);
    vkCmdCopyBufferToImage(cmd, buffer, color.image, kDst, 1, &region);
    Layout(cmd, color, kDst, kGeneral);
    VkClearColorValue zero = {};
    VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (Image *img : {&depth, &mv, &output}) {
        Layout(cmd, *img, VK_IMAGE_LAYOUT_UNDEFINED, kDst);
        vkCmdClearColorImage(cmd, img->image, kDst, &zero, 1, &all);
        Layout(cmd, *img, kDst, kGeneral);
    }
    submit();

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
    begin();
    r = SehCreate(b_create, create, cmd, p, &handle, &code);
    submit();
    std::printf("VULKAN_CreateFeature(18) %ux%u = 0x%08X seh=0x%08lX handle=%p\n", w, h, static_cast<unsigned>(r), code, handle);
    if (NVSDK_NGX_FAILED(r) || !handle) return 2;

    LARGE_INTEGER f0, f1, freq;
    QueryPerformanceFrequency(&freq);
    for (int f = 0; f < frames; ++f) {
        for (const char *name : {"Color", "DLSSNR.Color"}) p->Set(name, static_cast<void *>(&color.ngx));
        for (const char *name : {"Output", "DLSSNR.Output"}) p->Set(name, static_cast<void *>(&output.ngx));
        for (const char *name : {"Depth", "DLSSNR.Depth"}) p->Set(name, static_cast<void *>(&depth.ngx));
        for (const char *name : {"MotionVectors", "DLSSNR.MVec"}) p->Set(name, static_cast<void *>(&mv.ngx));
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
        begin();
        r = SehEvaluate(b_evaluate, evaluate, cmd, handle, p, &code);
        submit();
        QueryPerformanceCounter(&f1);
        std::printf("frame %d: VULKAN_EvaluateFeature = 0x%08X seh=0x%08lX  %.1f ms (including the wait for the GPU)\n", f,
            static_cast<unsigned>(r), code, (f1.QuadPart - f0.QuadPart) * 1000.0 / freq.QuadPart);
        if (NVSDK_NGX_FAILED(r)) return 3;
    }

    begin();
    Layout(cmd, output, kGeneral, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(cmd, output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    submit();
    FILE *fp = nullptr;
    _wfopen_s(&fp, out.c_str(), L"wb");
    if (!fp) { std::printf("cannot write the output file\n"); return 4; }
    std::fprintf(fp, "P6\n%u %u\n255\n", w, h);
    unsigned long long sum = 0, changed = 0;
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        std::fwrite(mapped + i * 4, 1, 3, fp);
        for (int c = 0; c < 3; ++c) {
            sum += mapped[i * 4 + c];
            changed += mapped[i * 4 + c] != pat[i * 4 + c];
        }
    }
    std::fclose(fp);
    std::printf("output mean %.2f, %.1f%% of values differ from the input\n", sum / (w * h * 3.0), changed * 100.0 / (w * h * 3.0));
    std::printf("VERDICT %s\n", sum == 0 ? "output is black" : (changed == 0 ? "output equals input" : "Vulkan path produced an image"));
    vkDeviceWaitIdle(g_dev);
    release(handle);
    return 0;
}
