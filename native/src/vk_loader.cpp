#include "vk_loader.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define MB_VK_DEFINE(name) PFN_##name name = nullptr;
MB_VK_GLOBAL_FUNCS(MB_VK_DEFINE)
MB_VK_INSTANCE_FUNCS(MB_VK_DEFINE)
MB_VK_DEVICE_FUNCS(MB_VK_DEFINE)
MB_VK_OPTIONAL_DEVICE_FUNCS(MB_VK_DEFINE)
MB_VK_OPTIONAL_INSTANCE_FUNCS(MB_VK_DEFINE)
#undef MB_VK_DEFINE

static PFN_vkGetInstanceProcAddr s_getInstanceProcAddr = nullptr;

namespace vkl {

bool loadLibrary(const char** err) {
    if (s_getInstanceProcAddr) return true;
#if defined(_WIN32)
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) { *err = "vulkan-1.dll"; return false; }
    s_getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(lib, "vkGetInstanceProcAddr")));
#else
    void* lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { *err = "libvulkan.so"; return false; }
    s_getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
#endif
    if (!s_getInstanceProcAddr) { *err = "vkGetInstanceProcAddr"; return false; }

    // vkEnumerateInstanceVersion is Vulkan 1.1+, tolerated as null (caller checks).
#define MB_LOAD(name) name = reinterpret_cast<PFN_##name>(s_getInstanceProcAddr(nullptr, #name));
    MB_VK_GLOBAL_FUNCS(MB_LOAD)
#undef MB_LOAD
    if (!vkCreateInstance) { *err = "vkCreateInstance"; return false; }
    return true;
}

bool loadInstance(VkInstance instance, const char** err) {
#define MB_LOAD(name)                                                                 \
    name = reinterpret_cast<PFN_##name>(s_getInstanceProcAddr(instance, #name));      \
    if (!name) { *err = #name; return false; }
    MB_VK_INSTANCE_FUNCS(MB_LOAD)
#undef MB_LOAD
#define MB_LOAD_OPT(name) name = reinterpret_cast<PFN_##name>(s_getInstanceProcAddr(instance, #name));
    MB_VK_OPTIONAL_INSTANCE_FUNCS(MB_LOAD_OPT)
#undef MB_LOAD_OPT
    return true;
}

bool loadDevice(VkDevice device, const char** err) {
#define MB_LOAD(name)                                                       \
    name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name)); \
    if (!name) { *err = #name; return false; }
    MB_VK_DEVICE_FUNCS(MB_LOAD)
#undef MB_LOAD
#define MB_LOAD_OPT(name) name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name));
    MB_VK_OPTIONAL_DEVICE_FUNCS(MB_LOAD_OPT)
#undef MB_LOAD_OPT
    return true;
}

}  // namespace vkl
