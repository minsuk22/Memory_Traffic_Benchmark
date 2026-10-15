#include "membench.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <stdexcept>
#include <thread>

#include "reference.h"
#include "vk_loader.h"

namespace mb {
namespace {

// SPIR-V generated at build time by glslc (see CMakeLists.txt).
const uint32_t kReadSpv[] = {
#include "read.spv.inc"
};
const uint32_t kWriteSpv[] = {
#include "write.spv.inc"
};
const uint32_t kCopySpv[] = {
#include "copy.spv.inc"
};
const uint32_t kVerifySpv[] = {
#include "verify.spv.inc"
};
const uint32_t kGemmTiledSpv[] = {
#include "gemm_tiled.spv.inc"
};
const uint32_t kGemmCoopF32Spv[] = {
#include "gemm_coop_f32.spv.inc"
};
const uint32_t kGemmCoopF16Spv[] = {
#include "gemm_coop_f16.spv.inc"
};

// Must match the push_constant block in common.glsl (std430).
struct Push {
    uint32_t chunkWordBase, logicalBase, linesMask, linesBits;
    uint32_t vplShift, pattern, seedMix, outSlot;
    uint32_t keyA[4];
    uint32_t keyB[4];
    uint32_t itersPerThread, numThreads, dstVecOffset, mode;
};
static_assert(sizeof(Push) == 80, "push constant layout must match common.glsl");

constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint32_t kSeq = 0, kRand = 1;

std::string fmt(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

#define VK_CHECK(expr)                                                                     \
    do {                                                                                   \
        VkResult r_ = (expr);                                                              \
        if (r_ != VK_SUCCESS)                                                              \
            throw std::runtime_error(fmt("%s failed: VkResult %d (line %d)", #expr, (int)r_, \
                                         __LINE__));                                       \
    } while (0)

uint32_t log2u(uint64_t v) {
    uint32_t r = 0;
    while ((2ull << r) <= v) ++r;
    return r;
}
uint64_t floorPow2(uint64_t v) { return v ? (1ull << log2u(v)) : 0; }
bool isPow2(uint64_t v) { return v && !(v & (v - 1)); }

const char* patName(uint32_t p) { return p == kRand ? "rand" : "seq"; }

// Per-chunk permutation key, shared by GPU (push constants) and CPU reference.
void deriveKey(uint32_t base, uint32_t chunk, uint32_t out[4]) {
    for (uint32_t i = 0; i < 4; ++i) out[i] = ref::hash32(base + 0x9E3779B9u * (chunk * 4u + i + 1u));
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Job {
    VkPipeline pipe     = VK_NULL_HANDLE;
    uint32_t linesBits  = 0;  // permutation domain = 2^linesBits lines (per chunk)
    uint32_t passes     = 1;  // every pass visits every line once, with a new permutation
    uint32_t pattern    = kRand;
    uint32_t seedMix    = 0;
    uint32_t keyBaseA   = 0;
    uint32_t keyBaseB   = 0;
    uint32_t dstVecOffset = 0;
    uint32_t mode       = 0;
    uint32_t numChunks  = 1;
    uint32_t unroll     = 4;  // iterations per thread must be a multiple of this
};

struct JobResult {
    double gpuMs = 0, cpuMs = 0;
    uint32_t chkX = 0, chkY = 0;
    uint32_t wgDone = 0, wgExpected = 0;
    double ms() const { return gpuMs > 0 ? gpuMs : cpuMs; }
};

struct Row {
    std::string name;
    std::vector<double> gbps;
    bool verified;
};

class Bench {
public:
    Bench(const Options& o, const LogFn& log) : o_(o), logFn_(log) {}
    ~Bench() { destroy(); }
    int run();

private:
    void log(const char* f, ...);
    void verdict(const char* level, const std::string& text);

    void initVulkan();
    void createResources();
    VkPipeline makePipeline(const uint32_t* code, size_t bytes, uint32_t sink);
    void createPipelines();  // for the current o_.wg
    void destroyPipelines();
    size_t allocChunks(size_t n);  // appends up to n chunks, returns how many were allocated
    void freeChunks();
    void printPipelineStats(VkPipeline p, const char* name);
    void destroy();

    uint64_t vecsPerChunk(const Job& j) const { return uint64_t(j.passes) << (j.linesBits + vplShift_); }
    double jobBytes(const Job& j) const { return double(vecsPerChunk(j)) * 16.0 * j.numChunks; }
    JobResult execute(const Job& j);
    void cpuReference(const Job& j, uint32_t out[2]);
    Job readJob(uint32_t pattern, uint32_t key) const;

    // One timed run: returns its time in ms and sets `line` to the per-run log text.
    using RunFn = std::function<double(int i, std::string& line)>;
    std::vector<double> measure(const RunFn& runOnce, double bytes, const std::function<bool()>& ready = {});
    void stability(const std::string& name, const std::vector<double>& gbps);

    void prewarm();
    void probe(const char* when);
    void boost();
    void setupGemm(bool coop);
    void destroyGemm();
    void fill(uint32_t seedMix);
    bool verifyContent(uint32_t seedMix, const char* what);
    void testRead(uint32_t pattern);
    void antiOptChecks();
    void testCopy(uint32_t pattern);
    void testWrite(uint32_t pattern);
    void readAfterWrite();
    void sweep();

    // Peak finder (--peak)
    struct Stat {
        double best = 0, med = 0;  // GB/s
        bool ok = true;            // all workgroups ran
        std::vector<double> runs;
    };
    void setConfig(uint32_t wg, uint32_t threads, uint32_t line);
    Stat timeJob(const Job& j, double bytes, int reps);
    std::pair<double, double> probeVals();  // READ seq (best of 3), READ rand (1 run) GB/s
    double cpuCopyGBps();
    void timeline(double ms);
    bool checkRead(uint32_t pattern, const char* what);
    void peak();
    int summary();

    Options o_;
    LogFn logFn_;

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties memProps_{};
    uint64_t maxAlloc_ = 0;
    uint32_t tsBits_ = 0;
    bool statsExt_ = false;

    struct Chunk {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };
    std::vector<Chunk> chunks_;
    uint64_t chunkBytes_ = 0;
    uint32_t chunkLinesBits_ = 0;
    uint32_t vplShift_ = 0;
    uint64_t threadCap_ = 0;
    int dataType_ = -1;

    VkBuffer outBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory outMem_ = VK_NULL_HANDLE;
    volatile uint32_t* outMap_ = nullptr;

    VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout pl_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkPipeline pRead_ = VK_NULL_HANDLE, pReadDead_ = VK_NULL_HANDLE, pWrite_ = VK_NULL_HANDLE,
               pCopy_ = VK_NULL_HANDLE, pVerify_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool qpool_ = VK_NULL_HANDLE;

    // Cooperative-matrix configuration for the boost GEMM (VK_KHR_cooperative_matrix).
    struct CoopCfg {
        bool avail = false;    // extension + features enabled + a usable fp16 configuration
        bool accF16 = false;   // fp16 accumulator (only if no fp32 one is offered)
        uint32_t M = 0, N = 0, K = 0, subgroup = 0;
    } coop_;
    // Boost GEMM resources (n x n, created on first use).
    struct Gemm {
        bool coop = false;
        VkBuffer buf[3] = {};
        VkDeviceMemory mem[3] = {};
        void* cMap = nullptr;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VkPipelineLayout pl = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkPipeline pipe = VK_NULL_HANDLE;
        VkCommandBuffer cmd[2] = {};
        VkFence fence[2] = {};
    } gemm_;
    static constexpr uint32_t kGemmN = 1024;

    uint32_t seed_ = 0;
    uint32_t curSeedMix_ = 0;  // seed of what the memory currently holds
    uint32_t readKey_ = 0;
    double randReadMs_ = 0;    // median of the main random read test
    uint32_t randReadChk_[2] = {0, 0};
    bool haveRandRead_ = false;

    std::vector<Row> rows_;
    std::vector<std::pair<std::string, std::string>> verdicts_;
};

void Bench::log(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    logFn_(buf);
}

void Bench::verdict(const char* level, const std::string& text) {
    verdicts_.emplace_back(level, text);
    log("  -> [%s] %s", level, text.c_str());
}

// ---------------------------------------------------------------------------------------
// Vulkan setup
// ---------------------------------------------------------------------------------------

void Bench::initVulkan() {
    const char* missing = nullptr;
    if (!vkl::loadLibrary(&missing)) throw std::runtime_error(fmt("cannot load Vulkan: %s", missing));
    uint32_t instVer = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&instVer);
    if (instVer < VK_API_VERSION_1_1) throw std::runtime_error("Vulkan 1.1 instance required");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "membench";
    app.apiVersion = VK_API_VERSION_1_3;  // effective version = min(this, device)
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst_));
    if (!vkl::loadInstance(inst_, &missing)) throw std::runtime_error(fmt("missing %s", missing));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst_, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst_, &n, devs.data());
    if (n == 0) throw std::runtime_error("no Vulkan device");
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        log("  device %u: %s%s", i, p.deviceName, (int)i == o_.device ? "  <- selected" : "");
    }
    if (o_.device < 0 || (uint32_t)o_.device >= n) throw std::runtime_error("bad --device index");
    phys_ = devs[o_.device];

    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, exts.data());
    auto hasExt = [&](const char* name) {
        for (auto& e : exts)
            if (!strcmp(e.extensionName, name)) return true;
        return false;
    };
    const bool hasStatsExt = hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    const bool hasCoopExt = hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties sgp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cmp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &m3;
    m3.pNext = &sgp;
    if (hasCoopExt) sgp.pNext = &cmp;
    vkGetPhysicalDeviceProperties2(phys_, &p2);
    props_ = p2.properties;
    maxAlloc_ = m3.maxMemoryAllocationSize;
    if (props_.apiVersion < VK_API_VERSION_1_1) throw std::runtime_error("Vulkan 1.1 device required");
    vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);

    // Compute queue, preferably one with timestamps.
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, qf.data());
    int best = -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (!(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
        if (best < 0 || (qf[best].timestampValidBits == 0 && qf[i].timestampValidBits > 0)) best = (int)i;
    }
    if (best < 0) throw std::runtime_error("no compute queue");
    qfam_ = (uint32_t)best;
    tsBits_ = qf[best].timestampValidBits;

    // Optional features: driver statistics of the compiled pipelines (instruction counts
    // etc.) and cooperative matrices + fp16 for the boost GEMM (needs a Vulkan 1.2 device).
    const bool v12 = props_.apiVersion >= VK_API_VERSION_1_2;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pef{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    {
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        void** tail = &f2.pNext;
        auto link = [&](auto* s) { *tail = s; tail = &s->pNext; };
        if (hasStatsExt) link(&pef);
        if (v12) { link(&f11); link(&f12); }
        if (v12 && hasCoopExt) link(&fcm);
        vkGetPhysicalDeviceFeatures2(phys_, &f2);
    }
    statsExt_ = hasStatsExt && pef.pipelineExecutableInfo;

    std::string coopWhy;
    if (!hasCoopExt) coopWhy = "VK_KHR_cooperative_matrix not supported";
    else if (!v12) coopWhy = "Vulkan 1.2 device required";
    else if (!fcm.cooperativeMatrix || !f12.shaderFloat16 || !f12.vulkanMemoryModel || !f11.storageBuffer16BitAccess)
        coopWhy = "required features missing (cooperativeMatrix, shaderFloat16, vulkanMemoryModel, 16-bit storage)";
    else if (!(cmp.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
        coopWhy = "not supported in compute shaders";
    else if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        coopWhy = "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR missing";
    else {
        // fp16 x fp16 with fp32 accumulation preferred, fp16 accumulation otherwise;
        // 16x16x16 preferred, any tile size that divides the matrix otherwise.
        uint32_t cnt = 0;
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(phys_, &cnt, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> cps(cnt, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(phys_, &cnt, cps.data());
        int bestScore = -1;
        for (auto& c : cps) {
            if (c.scope != VK_SCOPE_SUBGROUP_KHR || c.AType != VK_COMPONENT_TYPE_FLOAT16_KHR ||
                c.BType != VK_COMPONENT_TYPE_FLOAT16_KHR || c.CType != c.ResultType)
                continue;
            bool f32 = c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR;
            if (!f32 && c.CType != VK_COMPONENT_TYPE_FLOAT16_KHR) continue;
            if (!isPow2(c.MSize) || !isPow2(c.NSize) || !isPow2(c.KSize) || 2 * c.MSize > kGemmN ||
                2 * c.NSize > kGemmN || c.KSize > kGemmN)
                continue;
            int score = (f32 ? 2 : 0) + (c.MSize == 16 && c.NSize == 16 && c.KSize == 16 ? 1 : 0);
            if (score > bestScore) {
                bestScore = score;
                coop_.accF16 = !f32;
                coop_.M = c.MSize;
                coop_.N = c.NSize;
                coop_.K = c.KSize;
            }
        }
        coop_.subgroup = sgp.subgroupSize;
        if (bestScore < 0) coopWhy = "no fp16 subgroup configuration";
        else if (!coop_.subgroup || coop_.subgroup > props_.limits.maxComputeWorkGroupInvocations)
            coopWhy = "unusable subgroup size";
        else coop_.avail = true;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    std::vector<const char*> extNames;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR epef{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    void* chain = nullptr;
    void** tail = &chain;
    auto link = [&](auto* s) { *tail = s; tail = &s->pNext; };
    if (statsExt_) {
        extNames.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        epef.pipelineExecutableInfo = VK_TRUE;
        link(&epef);
    }
    if (coop_.avail) {
        extNames.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        e11.storageBuffer16BitAccess = VK_TRUE;
        e12.shaderFloat16 = VK_TRUE;
        e12.vulkanMemoryModel = VK_TRUE;
        ecm.cooperativeMatrix = VK_TRUE;
        link(&e11);
        link(&e12);
        link(&ecm);
    }
    dci.pNext = chain;
    dci.enabledExtensionCount = uint32_t(extNames.size());
    dci.ppEnabledExtensionNames = extNames.data();
    VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
    if (!vkl::loadDevice(dev_, &missing)) throw std::runtime_error(fmt("missing %s", missing));
    if (!vkGetPipelineExecutablePropertiesKHR || !vkGetPipelineExecutableStatisticsKHR) statsExt_ = false;
    vkGetDeviceQueue(dev_, qfam_, 0, &queue_);

    uint32_t dv = props_.driverVersion;
    log("GPU            : %s (vendor 0x%04x, device 0x%04x)", props_.deviceName, props_.vendorID, props_.deviceID);
    log("Vulkan API     : %u.%u.%u   driver version 0x%08x", VK_VERSION_MAJOR(props_.apiVersion),
        VK_VERSION_MINOR(props_.apiVersion), VK_VERSION_PATCH(props_.apiVersion), dv);
    log("Limits         : maxStorageBufferRange %u MiB, maxMemoryAllocationSize %llu MiB, maxWGInvocations %u",
        props_.limits.maxStorageBufferRange / (uint32_t)kMiB, (unsigned long long)(maxAlloc_ / kMiB),
        props_.limits.maxComputeWorkGroupInvocations);
    log("Timestamps     : %u valid bits, period %.3f ns", tsBits_, props_.limits.timestampPeriod);
    if (coop_.avail)
        log("Matrix units   : VK_KHR_cooperative_matrix, fp16 x fp16 -> %s, %ux%ux%u tiles, subgroup %u",
            coop_.accF16 ? "fp16" : "fp32", coop_.M, coop_.N, coop_.K, coop_.subgroup);
    else
        log("Matrix units   : not used (%s)", coopWhy.c_str());
    for (uint32_t i = 0; i < memProps_.memoryHeapCount; ++i)
        log("Memory heap %u  : %8llu MiB %s", i, (unsigned long long)(memProps_.memoryHeaps[i].size / kMiB),
            memProps_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? "DEVICE_LOCAL" : "");
}

int pickMemType(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, VkMemoryPropertyFlags want,
                VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return (int)i;
    }
    return -1;
}

std::string memFlagsStr(VkMemoryPropertyFlags f) {
    std::string s;
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) s += "DEVICE_LOCAL ";
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) s += "HOST_VISIBLE ";
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) s += "HOST_COHERENT ";
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) s += "HOST_CACHED ";
    return s.empty() ? "(none)" : s;
}

void Bench::createResources() {
    // --- sizes -----------------------------------------------------------------------
    if (!isPow2(o_.lineBytes) || o_.lineBytes < 16 || o_.lineBytes > 4096)
        throw std::runtime_error("--line must be a power of two in [16, 4096]");
    if (!isPow2(o_.wg) || o_.wg > 1024 /* MAX_WG in common.glsl */ ||
        o_.wg > props_.limits.maxComputeWorkGroupInvocations ||
        o_.wg > props_.limits.maxComputeWorkGroupSize[0])
        throw std::runtime_error("--wg must be a power of two within device limits");
    vplShift_ = log2u(o_.lineBytes / 16);

    const uint64_t total = o_.sizeMB * kMiB;
    uint64_t chunk = o_.chunkMB ? floorPow2(uint64_t(o_.chunkMB) * kMiB) : 256 * kMiB;
    chunk = std::min({chunk, floorPow2(props_.limits.maxStorageBufferRange), floorPow2(maxAlloc_),
                      floorPow2(total)});
    if (chunk < 64 * 1024) throw std::runtime_error("working set too small (min 64 KiB)");
    if (total >= 16ull * 1024 * kMiB) throw std::runtime_error("--size-mb must be < 16384");
    chunkBytes_ = chunk;
    chunkLinesBits_ = log2u(chunk / o_.lineBytes);
    uint64_t nChunks = std::min<uint64_t>(total / chunk, 64);

    threadCap_ = o_.threads ? floorPow2(o_.threads) : (1u << 18);
    threadCap_ = std::max<uint64_t>(threadCap_, o_.wg);

    // --- descriptor layout / pool ----------------------------------------------------
    VkDescriptorSetLayoutBinding b[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 2;
    dl.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &dsl_));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &pl_));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uint32_t(2 * nChunks)};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // chunks can be re-allocated
    dpci.maxSets = uint32_t(nChunks);
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &pool_));

    // --- result buffer (host visible, tiny) ------------------------------------------
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 256;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &outBuf_));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev_, outBuf_, &mr);
    int ot = pickMemType(memProps_, mr.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (ot < 0) throw std::runtime_error("no host-coherent memory");
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = (uint32_t)ot;
    VK_CHECK(vkAllocateMemory(dev_, &mai, nullptr, &outMem_));
    VK_CHECK(vkBindBufferMemory(dev_, outBuf_, outMem_, 0));
    void* p = nullptr;
    VK_CHECK(vkMapMemory(dev_, outMem_, 0, VK_WHOLE_SIZE, 0, &p));
    outMap_ = static_cast<volatile uint32_t*>(p);

    // --- data chunks -------------------------------------------------------------------
    size_t got = allocChunks(nChunks);
    if (got < nChunks)
        log("WARNING: allocation of chunk %zu failed; continuing with %zu chunk(s)", got, got);

    // --- command buffer, fence, timestamps -------------------------------------------
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = qfam_;
    VK_CHECK(vkCreateCommandPool(dev_, &cpci, nullptr, &cmdPool_));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdPool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &cbai, &cmd_));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &fence_));
    if (tsBits_) {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(dev_, &qpci, nullptr, &qpool_));
    }

    createPipelines();

    log("Data memory    : type %d [%s]", dataType_,
        memFlagsStr(memProps_.memoryTypes[dataType_].propertyFlags).c_str());
    log("Working set    : %llu MiB = %zu chunk(s) x %llu MiB  (requested %llu MiB)",
        (unsigned long long)(chunks_.size() * chunkBytes_ / kMiB), chunks_.size(),
        (unsigned long long)(chunkBytes_ / kMiB), (unsigned long long)o_.sizeMB);
    log("Access         : %u B per random address (%u lanes x 16 B), %llu lines per chunk",
        o_.lineBytes, 1u << vplShift_, (unsigned long long)(1ull << chunkLinesBits_));
    log("Dispatch       : workgroup %u, up to %llu invocations, 4 independent loads per iteration",
        o_.wg, (unsigned long long)threadCap_);
}

size_t Bench::allocChunks(size_t n) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = chunkBytes_;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    size_t got = 0;
    for (; got < n; ++got) {
        Chunk ch;
        VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &ch.buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev_, ch.buf, &mr);
        const VkMemoryPropertyFlags never =
            VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT;
        // Prefer plain device-local memory (no CPU-cached/snooped types).
        int t = pickMemType(memProps_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            never | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (t < 0) t = pickMemType(memProps_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, never);
        if (t < 0) t = pickMemType(memProps_, mr.memoryTypeBits, 0, never);
        if (t < 0) throw std::runtime_error("no usable memory type");
        dataType_ = t;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = (uint32_t)t;
        VkResult r = vkAllocateMemory(dev_, &mai, nullptr, &ch.mem);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(dev_, ch.buf, nullptr);
            if (chunks_.empty()) VK_CHECK(r);
            break;
        }
        VK_CHECK(vkBindBufferMemory(dev_, ch.buf, ch.mem, 0));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl_;
        VK_CHECK(vkAllocateDescriptorSets(dev_, &dai, &ch.set));
        VkDescriptorBufferInfo bi[2] = {{ch.buf, 0, VK_WHOLE_SIZE}, {outBuf_, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[2] = {};
        for (uint32_t i = 0; i < 2; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ch.set;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev_, 2, w, 0, nullptr);
        chunks_.push_back(ch);
    }
    return got;
}

void Bench::freeChunks() {
    vkDeviceWaitIdle(dev_);
    for (auto& c : chunks_) {
        vkFreeDescriptorSets(dev_, pool_, 1, &c.set);
        vkDestroyBuffer(dev_, c.buf, nullptr);
        vkFreeMemory(dev_, c.mem, nullptr);
    }
    chunks_.clear();
}

void Bench::createPipelines() {
    pRead_     = makePipeline(kReadSpv, sizeof kReadSpv, 1);
    pReadDead_ = makePipeline(kReadSpv, sizeof kReadSpv, 0);
    pWrite_    = makePipeline(kWriteSpv, sizeof kWriteSpv, 1);
    pCopy_     = makePipeline(kCopySpv, sizeof kCopySpv, 1);
    pVerify_   = makePipeline(kVerifySpv, sizeof kVerifySpv, 1);
}

void Bench::destroyPipelines() {
    vkDeviceWaitIdle(dev_);
    for (VkPipeline* p : {&pRead_, &pReadDead_, &pWrite_, &pCopy_, &pVerify_}) {
        if (*p) vkDestroyPipeline(dev_, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
}

VkPipeline Bench::makePipeline(const uint32_t* code, size_t bytes, uint32_t sink) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = bytes;
    smci.pCode = code;
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev_, &smci, nullptr, &sm));
    uint32_t spec[2] = {o_.wg, sink};
    VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
    VkSpecializationInfo si{2, me, sizeof spec, spec};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = &si;
    cpi.layout = pl_;
    if (statsExt_) cpi.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpi, nullptr, &p));
    vkDestroyShaderModule(dev_, sm, nullptr);
    return p;
}

void Bench::printPipelineStats(VkPipeline p, const char* name) {
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p;
    uint32_t n = 0;
    if (vkGetPipelineExecutablePropertiesKHR(dev_, &pi, &n, nullptr) != VK_SUCCESS) return;
    std::vector<VkPipelineExecutablePropertiesKHR> ex(n, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    vkGetPipelineExecutablePropertiesKHR(dev_, &pi, &n, ex.data());
    for (uint32_t e = 0; e < n; ++e) {
        VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = p;
        ei.executableIndex = e;
        uint32_t m = 0;
        vkGetPipelineExecutableStatisticsKHR(dev_, &ei, &m, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(m, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        vkGetPipelineExecutableStatisticsKHR(dev_, &ei, &m, st.data());
        std::string line = fmt("    %-22s [%s]", name, ex[e].name);
        for (auto& s : st) {
            switch (s.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                    line += fmt(" %s=%s", s.name, s.value.b32 ? "true" : "false"); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                    line += fmt(" %s=%lld", s.name, (long long)s.value.i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                    line += fmt(" %s=%llu", s.name, (unsigned long long)s.value.u64); break;
                default:
                    line += fmt(" %s=%.2f", s.name, s.value.f64); break;
            }
            line += ";";
        }
        log("%s", line.c_str());
    }
}

void Bench::destroy() {
    if (dev_) {
        destroyGemm();
        vkDeviceWaitIdle(dev_);
        for (VkPipeline p : {pRead_, pReadDead_, pWrite_, pCopy_, pVerify_})
            if (p) vkDestroyPipeline(dev_, p, nullptr);
        if (qpool_) vkDestroyQueryPool(dev_, qpool_, nullptr);
        if (fence_) vkDestroyFence(dev_, fence_, nullptr);
        if (cmdPool_) vkDestroyCommandPool(dev_, cmdPool_, nullptr);
        if (pool_) vkDestroyDescriptorPool(dev_, pool_, nullptr);
        if (pl_) vkDestroyPipelineLayout(dev_, pl_, nullptr);
        if (dsl_) vkDestroyDescriptorSetLayout(dev_, dsl_, nullptr);
        for (auto& c : chunks_) {
            vkDestroyBuffer(dev_, c.buf, nullptr);
            vkFreeMemory(dev_, c.mem, nullptr);
        }
        if (outBuf_) vkDestroyBuffer(dev_, outBuf_, nullptr);
        if (outMem_) vkFreeMemory(dev_, outMem_, nullptr);
        vkDestroyDevice(dev_, nullptr);
        dev_ = VK_NULL_HANDLE;
    }
    if (inst_) {
        vkDestroyInstance(inst_, nullptr);
        inst_ = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------------------

JobResult Bench::execute(const Job& j) {
    const uint64_t vecs = vecsPerChunk(j);
    if (vecs > 0xFFFFFFFFull) throw std::runtime_error("too many vectors per dispatch");
    const uint64_t T = std::min<uint64_t>(vecs / j.unroll, threadCap_);
    if (T < o_.wg) throw std::runtime_error("region too small for one workgroup");
    const uint32_t groups = uint32_t(T / o_.wg);
    uint32_t iters = uint32_t(vecs / T);
    // Self-test: emulate a driver that drops part of the work (the last loop iterations).
    if (o_.injectFault && j.pipe != pVerify_ && j.pipe != pReadDead_ && iters > j.unroll) iters -= j.unroll;

    for (int i = 0; i < 4; ++i) outMap_[i] = 0;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandBuffer(cmd_, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));

    // Make previous submissions' writes (and the host zeroing) visible.
    VkMemoryBarrier mbar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mbar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    mbar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mbar, 0, nullptr, 0, nullptr);

    if (qpool_) {
        vkCmdResetQueryPool(cmd_, qpool_, 0, 2);
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool_, 0);
    }
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, j.pipe);
    for (uint32_t c = 0; c < j.numChunks; ++c) {
        Push pc{};
        pc.chunkWordBase = uint32_t(c * (chunkBytes_ / 4));
        pc.logicalBase = uint32_t(c * vecs);
        pc.linesMask = (1u << j.linesBits) - 1u;
        pc.linesBits = j.linesBits;
        pc.vplShift = vplShift_;
        pc.pattern = j.pattern;
        pc.seedMix = j.seedMix;
        pc.outSlot = 0;
        deriveKey(j.keyBaseA, c, pc.keyA);
        deriveKey(j.keyBaseB, c, pc.keyB);
        pc.itersPerThread = iters;
        pc.numThreads = uint32_t(T);
        pc.dstVecOffset = j.dstVecOffset;
        pc.mode = j.mode;
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pl_, 0, 1, &chunks_[c].set, 0, nullptr);
        vkCmdPushConstants(cmd_, pl_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cmd_, groups, 1, 1);
    }
    if (qpool_) vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, 1);

    mbar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mbar.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mbar,
                         0, nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(cmd_));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VK_CHECK(vkResetFences(dev_, 1, &fence_));
    auto t0 = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
    VK_CHECK(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, 120ull * 1000000000ull));
    auto t1 = std::chrono::steady_clock::now();

    JobResult r;
    r.cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (qpool_) {
        uint64_t ts[2] = {0, 0};
        VK_CHECK(vkGetQueryPoolResults(dev_, qpool_, 0, 2, sizeof ts, ts, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        uint64_t mask = tsBits_ >= 64 ? ~0ull : ((1ull << tsBits_) - 1);
        r.gpuMs = double((ts[1] - ts[0]) & mask) * props_.limits.timestampPeriod / 1e6;
    }
    r.chkX = outMap_[0];
    r.chkY = outMap_[1];
    r.wgDone = outMap_[2];
    r.wgExpected = groups * j.numChunks;
    return r;
}

// Same checksum as read.comp, computed on all CPU cores.
void Bench::cpuReference(const Job& j, uint32_t out[2]) {
    const uint64_t vecs = vecsPerChunk(j);
    const uint64_t piece = 1u << 20;
    struct Task { uint32_t c, b, e; };
    std::vector<Task> tasks;
    for (uint32_t c = 0; c < j.numChunks; ++c)
        for (uint64_t b = 0; b < vecs; b += piece)
            tasks.push_back({c, uint32_t(b), uint32_t(std::min(vecs, b + piece))});

    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<size_t> next{0};
    std::vector<std::array<uint32_t, 2>> part(nt, {0u, 0u});
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            for (;;) {
                size_t i = next++;
                if (i >= tasks.size()) break;
                const Task& k = tasks[i];
                ref::Params p{(1u << j.linesBits) - 1u, j.linesBits, vplShift_, j.pattern, j.seedMix, {}};
                deriveKey(j.keyBaseA, k.c, p.keyA);
                ref::readChecksum(p, uint32_t(k.c * (chunkBytes_ / 4)), uint32_t(k.c * vecs), k.b, k.e,
                                  part[t].data());
            }
        });
    }
    for (auto& t : th) t.join();
    out[0] = out[1] = 0;
    for (auto& p : part) { out[0] += p[0]; out[1] += p[1]; }
}

Job Bench::readJob(uint32_t pattern, uint32_t key) const {
    Job j;
    j.pipe = pRead_;
    j.linesBits = chunkLinesBits_;
    j.pattern = pattern;
    j.seedMix = curSeedMix_;
    j.keyBaseA = key;
    j.numChunks = uint32_t(chunks_.size());
    return j;
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

// Warm-up, then o_.reps measured runs. Warm-up lasts at least o_.warmup runs and
// o_.warmupMs of GPU time, so short kernels (sequential: ~20 ms) still give the GPU and
// DRAM clocks time to settle, and continues while `ready` is false (e.g. the CPU
// reference still running). Returns GB/s of the measured runs.
std::vector<double> Bench::measure(const RunFn& runOnce, double bytes, const std::function<bool()>& ready) {
    const int kMaxWarm = 1000;
    std::vector<double> gbps;
    int warmRuns = 0, extra = 0;
    double warmMs = 0;
    std::string extraLine;
    for (int i = 0;; ++i) {
        bool warm = warmRuns < o_.warmup ||
                    (warmRuns < kMaxWarm && (warmMs < o_.warmupMs || (ready && !ready())));
        if (!warm && int(gbps.size()) >= o_.reps) break;
        std::string line;
        double ms = runOnce(i, line);
        if (warm) {
            ++warmRuns;
            warmMs += ms;
            if (warmRuns <= o_.warmup) {
                log("  warm %d: %s", warmRuns, line.c_str());
            } else {
                ++extra;
                extraLine = line;
            }
            continue;
        }
        if (extra) {
            log("  warm +%d runs (warm-up total: %d runs, %.0f ms GPU time), last: %s", extra, warmRuns, warmMs,
                extraLine.c_str());
            extra = 0;
        }
        gbps.push_back(bytes / (ms * 1e6));
        log("  rep  %d: %s", int(gbps.size()), line.c_str());
    }
    return gbps;
}

// Large run-to-run spread means the clocks changed during the measurement (DVFS,
// thermal throttling, background load): the numbers are correct but not stable.
void Bench::stability(const std::string& name, const std::vector<double>& gbps) {
    if (gbps.size() < 2) return;
    auto mm = std::minmax_element(gbps.begin(), gbps.end());
    double spread = (*mm.second - *mm.first) / median(gbps);
    if (spread > 0.10)
        verdict("WARN", fmt("%s: run-to-run spread %.0f%% - clocks or temperature changed during the "
                            "measurement, re-run (cool, plugged in, screen on)", name.c_str(), 100.0 * spread));
}

// Keeps the GPU and DRAM busy before the first measurement so DVFS can raise the
// clocks. Uses another permutation key than the tests, so whatever it leaves in the
// caches (at most a few MiB of a much larger set) is unrelated to the first test.
void Bench::prewarm() {
    if (o_.prewarmMs <= 0) return;
    Job seq = readJob(kSeq, 0);
    Job rnd = readJob(kRand, readKey_ ^ 0x3C6EF372u);
    auto t0 = std::chrono::steady_clock::now();
    double ms = 0, gb = 0;
    int n = 0;
    while (ms < o_.prewarmMs) {
        const Job& j = (n % 4 == 3) ? rnd : seq;  // mostly sequential: high bandwidth raises DRAM clocks
        execute(j);
        gb += jobBytes(j) / 1e9;
        ++n;
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    log("");
    log("[PRE-WARM] %d read runs, %.1f GB in %.0f ms to raise GPU/DRAM clocks (separate key, not measured)", n,
        gb, ms);
}

// Short read measurement used to see the effect of the boost on the memory clocks.
void Bench::probe(const char* when) {
    Job s = readJob(kSeq, 0);
    Job r = readJob(kRand, readKey_ ^ 0x9E3779B9u);
    double sMs = 1e300;
    for (int i = 0; i < 3; ++i) sMs = std::min(sMs, execute(s).ms());
    double rMs = execute(r).ms();
    log("  probe %-12s: READ seq %6.2f GB/s, READ rand %5.2f GB/s", when, jobBytes(s) / (sMs * 1e6),
        jobBytes(r) / (rMs * 1e6));
}

// Buffers, pipeline and command buffers for the n x n boost GEMM. A and B are filled
// with 1.0, so every element of C must be exactly n.
void Bench::setupGemm(bool coop) {
    destroyGemm();
    Gemm& g = gemm_;
    g.coop = coop;
    const uint64_t nn = uint64_t(kGemmN) * kGemmN;
    const uint64_t inBytes = nn * (coop ? 2 : 4);
    const uint64_t outBytes = nn * (coop && coop_.accF16 ? 2 : 4);
    for (int i = 0; i < 3; ++i) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = i < 2 ? inBytes : outBytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &g.buf[i]));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev_, g.buf[i], &mr);
        // C is host visible so the result can be checked.
        int t = i < 2 ? pickMemType(memProps_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)
                      : pickMemType(memProps_, mr.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        if (t < 0) t = pickMemType(memProps_, mr.memoryTypeBits, 0, 0);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = uint32_t(t);
        VK_CHECK(vkAllocateMemory(dev_, &mai, nullptr, &g.mem[i]));
        VK_CHECK(vkBindBufferMemory(dev_, g.buf[i], g.mem[i], 0));
    }
    VK_CHECK(vkMapMemory(dev_, g.mem[2], 0, VK_WHOLE_SIZE, 0, &g.cMap));

    VkDescriptorSetLayoutBinding b[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 3;
    dl.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &g.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &g.pl));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &g.pool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g.pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &g.dsl;
    VK_CHECK(vkAllocateDescriptorSets(dev_, &dai, &g.set));
    VkDescriptorBufferInfo bi[3];
    VkWriteDescriptorSet w[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        bi[i] = {g.buf[i], 0, VK_WHOLE_SIZE};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = g.set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev_, 3, w, 0, nullptr);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    if (coop) {
        smci.codeSize = coop_.accF16 ? sizeof kGemmCoopF16Spv : sizeof kGemmCoopF32Spv;
        smci.pCode = coop_.accF16 ? kGemmCoopF16Spv : kGemmCoopF32Spv;
    } else {
        smci.codeSize = sizeof kGemmTiledSpv;
        smci.pCode = kGemmTiledSpv;
    }
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev_, &smci, nullptr, &sm));
    uint32_t spec[4] = {coop_.subgroup, coop_.M, coop_.N, coop_.K};
    VkSpecializationMapEntry me[4] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
    VkSpecializationInfo si{4, me, sizeof spec, spec};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = coop ? &si : nullptr;
    cpi.layout = g.pl;
    VkResult r = vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpi, nullptr, &g.pipe);
    vkDestroyShaderModule(dev_, sm, nullptr);
    VK_CHECK(r);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdPool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &cbai, g.cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto& f : g.fence) VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &f));
}

void Bench::destroyGemm() {
    Gemm& g = gemm_;
    if (!dev_) return;
    vkDeviceWaitIdle(dev_);
    for (auto& f : g.fence)
        if (f) vkDestroyFence(dev_, f, nullptr);
    if (g.cmd[0]) vkFreeCommandBuffers(dev_, cmdPool_, 2, g.cmd);
    if (g.pipe) vkDestroyPipeline(dev_, g.pipe, nullptr);
    if (g.pool) vkDestroyDescriptorPool(dev_, g.pool, nullptr);
    if (g.pl) vkDestroyPipelineLayout(dev_, g.pl, nullptr);
    if (g.dsl) vkDestroyDescriptorSetLayout(dev_, g.dsl, nullptr);
    for (int i = 0; i < 3; ++i) {
        if (g.buf[i]) vkDestroyBuffer(dev_, g.buf[i], nullptr);
        if (g.mem[i]) vkFreeMemory(dev_, g.mem[i], nullptr);
    }
    g = Gemm{};
}

// Loads CPU and GPU right before the measurements so that DVFS raises the CPU, GPU and
// memory-interface clocks: all CPU cores run integer hashing, the GPU runs back-to-back
// 1024^3 GEMMs (matrix units via VK_KHR_cooperative_matrix when available). The probes
// before and after show whether this changes the memory bandwidth.
void Bench::boost() {
    if (o_.boostMs <= 0 || (!o_.boostCpu && !o_.boostGpu)) return;
    log("");
    log("[BOOST] %.0f ms of %s load right before the tests", o_.boostMs,
        o_.boostCpu && o_.boostGpu ? "CPU + GPU" : o_.boostCpu ? "CPU" : "GPU");
    probe("before boost");

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> cpuHashes{0};
    std::vector<std::thread> th;
    struct Joiner {  // stops the CPU load also when the GPU part throws
        std::atomic<bool>& stop;
        std::vector<std::thread>& th;
        ~Joiner() {
            stop = true;
            for (auto& t : th)
                if (t.joinable()) t.join();
        }
    } joiner{stop, th};
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (o_.boostCpu)
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([&, t] {
                uint32_t x = t + 1;
                uint64_t cnt = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    for (int i = 0; i < 4096; ++i) x = ref::hash32(x);
                    cnt += 4096;
                }
                cpuHashes += cnt + (x == 0);  // keep x alive
            });

    auto t0 = std::chrono::steady_clock::now();
    auto elapsedMs = [&] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    std::string gpuLine;
    if (o_.boostGpu) {
        bool coop = coop_.avail;
        for (;;) {
            try {
                setupGemm(coop);
                break;
            } catch (const std::exception& e) {
                if (!coop) throw;
                log("  cooperative-matrix GEMM unavailable (%s), using the fp32 shader GEMM", e.what());
                coop = false;
            }
        }
        Gemm& g = gemm_;
        const uint32_t n = kGemmN;
        const uint32_t groupsX = coop ? (n / (2 * coop_.M)) * (n / (2 * coop_.N)) : n / 16;
        const uint32_t groupsY = coop ? 1 : n / 16;
        auto record = [&](VkCommandBuffer cb, uint32_t count, bool init) {
            VK_CHECK(vkResetCommandBuffer(cb, 0));
            VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            if (init) {
                const uint32_t one = coop ? 0x3C003C00u : 0x3F800000u;  // 1.0 in fp16 pairs / fp32
                vkCmdFillBuffer(cb, g.buf[0], 0, VK_WHOLE_SIZE, one);
                vkCmdFillBuffer(cb, g.buf[1], 0, VK_WHOLE_SIZE, one);
                vkCmdFillBuffer(cb, g.buf[2], 0, VK_WHOLE_SIZE, 0);
                mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                     &mb, 0, nullptr, 0, nullptr);
            }
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.set, 0, nullptr);
            vkCmdPushConstants(cb, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
            for (uint32_t i = 0; i < count; ++i) {
                vkCmdDispatch(cb, groupsX, groupsY, 1);
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &mb, 0, nullptr, 0, nullptr);
            }
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                                 nullptr, 0, nullptr);
            VK_CHECK(vkEndCommandBuffer(cb));
        };
        auto submit = [&](int i) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &g.cmd[i];
            VK_CHECK(vkResetFences(dev_, 1, &g.fence[i]));
            VK_CHECK(vkQueueSubmit(queue_, 1, &si, g.fence[i]));
        };
        auto wait = [&](int i) { VK_CHECK(vkWaitForFences(dev_, 1, &g.fence[i], VK_TRUE, ~0ull)); };

        // One GEMM (plus buffer init) to size the batches to ~40 ms per submission.
        record(g.cmd[0], 1, true);
        auto s0 = std::chrono::steady_clock::now();
        submit(0);
        wait(0);
        double oneMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
        uint32_t batch = uint32_t(std::clamp(40.0 / std::max(oneMs, 0.01), 1.0, 256.0));
        record(g.cmd[0], batch, false);
        record(g.cmd[1], batch, false);

        // Two submissions in flight, so the GPU never waits for the CPU.
        uint64_t gemms = 1;
        auto g0 = std::chrono::steady_clock::now();
        submit(0);
        submit(1);
        int cur = 0;
        while (elapsedMs() < o_.boostMs) {
            wait(cur);
            gemms += batch;
            submit(cur);
            cur ^= 1;
        }
        wait(0);
        wait(1);
        gemms += 2ull * batch;
        double gMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
        double tflops = double(gemms - 1) * 2.0 * n * n * n / (gMs * 1e9);

        // Every element of C must be n (A = B = 1.0).
        uint64_t bad = 0;
        const uint64_t nn = uint64_t(n) * n;
        if (coop && coop_.accF16) {
            const uint16_t* c = static_cast<const uint16_t*>(g.cMap);
            for (uint64_t i = 0; i < nn; ++i) bad += c[i] != 0x6400;  // 1024.0 in fp16
        } else {
            const float* c = static_cast<const float*>(g.cMap);
            for (uint64_t i = 0; i < nn; ++i) bad += c[i] != float(n);
        }
        gpuLine = fmt("%s GEMM %ux%ux%u: %llu GEMMs (%u per submission), %.2f TFLOPS, result %s",
                      coop ? (coop_.accF16 ? "coopmat fp16->fp16" : "coopmat fp16->fp32") : "fp32 shader", n, n, n,
                      (unsigned long long)gemms, batch, tflops,
                      bad ? fmt("WRONG (%llu of %llu elements)", (unsigned long long)bad, (unsigned long long)nn).c_str()
                          : "correct");
        destroyGemm();
        if (bad)
            verdict("WARN", "boost GEMM produced a wrong result (driver issue?); it is only used as load, the "
                            "memory results are not affected");
    } else {
        while (elapsedMs() < o_.boostMs) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    stop = true;
    for (auto& t : th) t.join();
    th.clear();
    double ms = elapsedMs();
    if (o_.boostGpu) log("  GPU: %s", gpuLine.c_str());
    if (o_.boostCpu) log("  CPU: %u threads, %.0f M hashes/s", nt, double(cpuHashes.load()) / (ms * 1e3));
    log("  boost took %.0f ms", ms);
    probe("after boost");
}

void Bench::fill(uint32_t seedMix) {
    Job j;
    j.pipe = pWrite_;
    j.linesBits = chunkLinesBits_;
    j.pattern = kSeq;
    j.seedMix = seedMix;
    j.numChunks = uint32_t(chunks_.size());
    JobResult r = execute(j);
    curSeedMix_ = seedMix;
    log("  fill (sequential write) : %8.3f ms  %7.2f GB/s", r.ms(), jobBytes(j) / (r.ms() * 1e6));
}

bool Bench::verifyContent(uint32_t seedMix, const char* what) {
    Job v;
    v.pipe = pVerify_;
    v.linesBits = chunkLinesBits_;
    v.pattern = kSeq;
    v.seedMix = seedMix;
    v.numChunks = uint32_t(chunks_.size());
    v.unroll = 1;
    JobResult r = execute(v);
    bool ok = r.chkX == 0 && r.wgDone == r.wgExpected;
    if (!ok)
        log("  verify %s: %u mismatching 16-byte words, %u/%u workgroups", what, r.chkX, r.wgDone, r.wgExpected);
    return ok;
}

void Bench::testRead(uint32_t pattern) {
    Job j = readJob(pattern, readKey_);
    const double bytes = jobBytes(j);
    std::string name = fmt("READ  %-4s %4uB", patName(pattern), o_.lineBytes);
    log("");
    log("[%s] %.0f MiB, every line exactly once per run", name.c_str(), bytes / kMiB);

    // The CPU reference is computed while the GPU warms up (the GPU would otherwise idle
    // and drop its clocks). Measured runs only start once it is done, so it does not
    // compete with them; warm-up runs are compared against it afterwards.
    uint32_t refc[2] = {0, 0};
    double refMs = 0;
    std::future<void> refJob;
    if (o_.cpuRef)
        refJob = std::async(std::launch::async, [&] {
            auto t0 = std::chrono::steady_clock::now();
            cpuReference(j, refc);
            refMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        });
    auto refReady = [&] {
        return !refJob.valid() || refJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    };

    std::vector<double> msList;
    std::vector<std::array<uint32_t, 2>> chks;
    bool allWg = true;
    auto gbps = measure(
        [&](int, std::string& line) {
            JobResult r = execute(j);
            bool wgOk = r.wgDone == r.wgExpected;
            allWg &= wgOk;
            chks.push_back({r.chkX, r.chkY});
            const char* status = !wgOk ? "MISSING WORKGROUPS"
                                 : !o_.cpuRef ? "ok"
                                 : !refReady() ? "(CPU reference pending)"
                                 : (r.chkX == refc[0] && r.chkY == refc[1]) ? "MATCH" : "MISMATCH";
            line = fmt("%9.3f ms  %8.2f GB/s  chk %08x:%08x  wg %u/%u  %s", r.ms(), bytes / (r.ms() * 1e6), r.chkX,
                       r.chkY, r.wgDone, r.wgExpected, status);
            msList.push_back(r.ms());
            return r.ms();
        },
        bytes, refReady);
    msList.erase(msList.begin(), msList.end() - gbps.size());  // keep measured runs only

    bool allOk = allWg, allSame = true;
    for (auto& c : chks) allSame &= c == chks[0];
    if (o_.cpuRef) {
        refJob.get();
        for (auto& c : chks) allOk &= c[0] == refc[0] && c[1] == refc[1];
        log("  CPU reference checksum : %08x:%08x  (%.0f ms on %u threads, during warm-up)", refc[0], refc[1],
            refMs, std::max(1u, std::thread::hardware_concurrency()));
    }
    uint32_t first[2] = {chks[0][0], chks[0][1]};
    const int runs = int(chks.size());
    if (o_.cpuRef)
        verdict(allOk ? "PASS" : "FAIL",
                fmt("%s: GPU checksum %s CPU reference (%d runs)", name.c_str(),
                    allOk ? "equals" : "DIFFERS FROM", runs));
    verdict(allSame ? "PASS" : "FAIL",
            fmt("%s: checksum %s across repeated runs", name.c_str(), allSame ? "identical" : "NOT identical"));
    rows_.push_back({name, gbps, allOk && allSame});
    stability(name, gbps);
    if (pattern == kRand) {
        randReadMs_ = median(msList);
        randReadChk_[0] = first[0];
        randReadChk_[1] = first[1];
        haveRandRead_ = true;
    }
}

void Bench::antiOptChecks() {
    if (!haveRandRead_) return;
    log("");
    log("[ANTI-OPTIMISATION CHECKS] (random read, %.0f MiB)", jobBytes(readJob(kRand, readKey_)) / kMiB);

    // 1) Different permutation key -> must give a different checksum that still matches
    //    the CPU. A result that was cached / constant-folded would not change.
    {
        Job j = readJob(kRand, readKey_ ^ 0x5A5A1234u);
        JobResult r = execute(j);
        bool differs = r.chkX != randReadChk_[0] || r.chkY != randReadChk_[1];
        bool refOk = true;
        if (o_.cpuRef) {
            uint32_t refc[2];
            cpuReference(j, refc);
            refOk = r.chkX == refc[0] && r.chkY == refc[1];
        }
        log("  1. other key    : chk %08x:%08x (main %08x:%08x)", r.chkX, r.chkY, randReadChk_[0], randReadChk_[1]);
        verdict(differs && refOk ? "PASS" : "FAIL",
                fmt("key sensitivity: new key gives a %s checksum%s", differs ? "different" : "SAME",
                    o_.cpuRef ? (refOk ? " that matches the CPU" : " that does NOT match the CPU") : ""));
    }

    // 2) Dead-code control: same kernel, result never stored (SINK=0). If the driver
    //    removes unused loads, this collapses to ~0 ms while the real kernel cannot.
    {
        Job j = readJob(kRand, readKey_);
        j.pipe = pReadDead_;
        std::vector<double> ms;
        for (int i = 0; i < 3; ++i) ms.push_back(execute(j).ms());
        double dead = median(ms);
        double ratio = dead / randReadMs_;
        log("  2. dead control : %.3f ms vs real %.3f ms  (%.1f%%)", dead, randReadMs_, 100.0 * ratio);
        if (ratio < 0.25)
            verdict("PASS", fmt("driver deletes unused loads (control %.1f%% of real time) - the real kernel's "
                                "loads are therefore demonstrably NOT deleted", 100.0 * ratio));
        else
            verdict("INFO", fmt("driver did not delete the unused loads in the control (%.0f%% of real time); "
                                "the checksum match is the proof that loads executed", 100.0 * ratio));
    }

    // 3) Work scaling: 2 passes over the same set (new permutation each pass) must take
    //    about twice as long; skipped or merged work would not scale.
    {
        Job j1 = readJob(kRand, readKey_);
        Job j2 = j1;
        j2.passes = 2;
        std::vector<double> a, b;
        execute(j1);  // the GPU idled during the CPU reference above
        for (int i = 0; i < 3; ++i) { a.push_back(execute(j1).ms()); b.push_back(execute(j2).ms()); }
        JobResult r2 = execute(j2);
        bool refOk = true;
        if (o_.cpuRef) {
            uint32_t refc[2];
            cpuReference(j2, refc);
            refOk = r2.chkX == refc[0] && r2.chkY == refc[1];
        }
        double ratio = median(b) / median(a);
        log("  3. scaling      : 1 pass %.3f ms, 2 passes %.3f ms  -> x%.2f", median(a), median(b), ratio);
        bool okRatio = ratio > 1.7 && ratio < 2.3;
        verdict(okRatio && refOk ? "PASS" : (refOk ? "WARN" : "FAIL"),
                fmt("work scaling x%.2f for 2x traffic (expected ~2.0)%s", ratio,
                    refOk ? "" : ", 2-pass checksum does NOT match the CPU"));
    }

    // 4) Compute headroom: same kernel on a cache-resident footprint. If that is not much
    //    faster than the DRAM run, the kernel is ALU/issue bound and the DRAM number is a
    //    lower bound rather than the memory limit.
    {
        Job j = readJob(kRand, readKey_);
        uint64_t fp = std::min<uint64_t>(256 * 1024, chunkBytes_);
        j.linesBits = log2u(fp / o_.lineBytes);
        j.numChunks = 1;
        j.passes = uint32_t(std::max<uint64_t>(1, uint64_t(jobBytes(readJob(kRand, readKey_))) / fp));
        while (vecsPerChunk(j) > 0xFFFFFFFFull) j.passes /= 2;
        std::vector<double> g;
        for (int i = 0; i < 3; ++i) g.push_back(jobBytes(j) / (execute(j).ms() * 1e6));
        double cached = median(g);
        double dram = jobBytes(readJob(kRand, readKey_)) / (randReadMs_ * 1e6);
        log("  4. headroom     : %llu KiB footprint %.2f GB/s vs DRAM %.2f GB/s  (x%.1f)",
            (unsigned long long)(fp / 1024), cached, dram, cached / dram);
        verdict(cached > 1.5 * dram ? "PASS" : "WARN",
                fmt("kernel is %s (cache-resident run x%.1f faster than DRAM run)",
                    cached > 1.5 * dram ? "memory bound, not ALU bound" : "possibly ALU/issue bound", cached / dram));
    }

    if (statsExt_) {
        log("  5. driver pipeline statistics (VK_KHR_pipeline_executable_properties):");
        printPipelineStats(pRead_, "read (real)");
        printPipelineStats(pReadDead_, "read (dead control)");
    } else {
        log("  5. driver pipeline statistics: not supported by this driver");
    }
}

void Bench::testCopy(uint32_t pattern) {
    Job j;
    j.pipe = pCopy_;
    j.linesBits = chunkLinesBits_ - 1;  // lower half = source, upper half = destination
    j.pattern = pattern;
    j.seedMix = curSeedMix_;
    j.dstVecOffset = 1u << (chunkLinesBits_ - 1 + vplShift_);
    j.numChunks = uint32_t(chunks_.size());
    const double bytes = 2.0 * jobBytes(j);  // read + write
    std::string name = fmt("COPY  %-4s %4uB", patName(pattern), o_.lineBytes);
    log("");
    log("[%s] %.0f MiB read + %.0f MiB write, verified after every run", name.c_str(), bytes / 2 / kMiB,
        bytes / 2 / kMiB);

    bool allOk = true;
    auto gbps = measure(
        [&](int i, std::string& line) {
            // New source/destination permutations every run: a stale result from the
            // previous run would fail verification.
            j.keyBaseA = ref::hash32(seed_ ^ 0xC0C0u) + uint32_t(i);
            j.keyBaseB = ref::hash32(seed_ ^ 0xD5D5u) + uint32_t(i);
            JobResult r = execute(j);
            Job v = j;
            v.pipe = pVerify_;
            v.mode = 1;
            v.unroll = 1;
            JobResult rv = execute(v);
            bool ok = r.wgDone == r.wgExpected && rv.chkX == 0 && rv.wgDone == rv.wgExpected;
            allOk &= ok;
            line = fmt("%9.3f ms  %8.2f GB/s  verify: %u bad  %s", r.ms(), bytes / (r.ms() * 1e6), rv.chkX,
                       ok ? "OK" : "FAIL");
            return r.ms();
        },
        bytes);
    verdict(allOk ? "PASS" : "FAIL", fmt("%s: destination content %s", name.c_str(),
                                         allOk ? "verified in every run" : "WRONG in at least one run"));
    rows_.push_back({name, gbps, allOk});
    stability(name, gbps);
    fill(curSeedMix_);  // restore the destination halves
}

void Bench::testWrite(uint32_t pattern) {
    Job j;
    j.pipe = pWrite_;
    j.linesBits = chunkLinesBits_;
    j.pattern = pattern;
    j.numChunks = uint32_t(chunks_.size());
    const double bytes = jobBytes(j);
    std::string name = fmt("WRITE %-4s %4uB", patName(pattern), o_.lineBytes);
    log("");
    log("[%s] %.0f MiB, new data seed every run, full content verified after every run", name.c_str(),
        bytes / kMiB);

    bool allOk = true;
    auto gbps = measure(
        [&](int i, std::string& line) {
            j.seedMix = ref::hash32(seed_ ^ (0x57000000u + (pattern << 16) + uint32_t(i)));
            j.keyBaseA = ref::hash32(seed_ ^ 0xBEEFu) + uint32_t(i);
            JobResult r = execute(j);
            bool ok = r.wgDone == r.wgExpected && verifyContent(j.seedMix, name.c_str());
            curSeedMix_ = j.seedMix;
            allOk &= ok;
            line = fmt("%9.3f ms  %8.2f GB/s  seed %08x  %s", r.ms(), bytes / (r.ms() * 1e6), j.seedMix,
                       ok ? "OK" : "FAIL");
            return r.ms();
        },
        bytes);
    verdict(allOk ? "PASS" : "FAIL",
            fmt("%s: memory content %s", name.c_str(), allOk ? "verified after every run" : "WRONG after at least one run"));
    rows_.push_back({name, gbps, allOk});
    stability(name, gbps);
}

// Independent path: read the memory produced by the last write run with the random
// read kernel and compare against the CPU.
void Bench::readAfterWrite() {
    if (!o_.cpuRef) return;
    Job j = readJob(kRand, readKey_ ^ 0x77u);
    JobResult r = execute(j);
    uint32_t refc[2];
    cpuReference(j, refc);
    bool ok = r.chkX == refc[0] && r.chkY == refc[1] && r.wgDone == r.wgExpected;
    log("");
    log("[READ-AFTER-WRITE] chk %08x:%08x, CPU %08x:%08x", r.chkX, r.chkY, refc[0], refc[1]);
    verdict(ok ? "PASS" : "FAIL", "random read of the last write's data matches the CPU reference");
}

void Bench::sweep() {
    log("");
    log("[WORKING-SET SWEEP] random read, %u B lines, >= 1 GiB traffic per point (best of 3)", o_.lineBytes);
    log("  %12s  %10s", "footprint", "GB/s");
    for (uint64_t fp = 64 * 1024; fp <= chunkBytes_; fp *= 2) {
        Job j = readJob(kRand, readKey_);
        j.numChunks = 1;
        j.linesBits = log2u(fp / o_.lineBytes);
        j.passes = uint32_t(std::max<uint64_t>(1, (1024 * kMiB) / fp));
        while (vecsPerChunk(j) > 0xFFFFFFFFull) j.passes /= 2;
        execute(j);  // warm-up
        double best = 0;
        for (int i = 0; i < 3; ++i) best = std::max(best, jobBytes(j) / (execute(j).ms() * 1e6));
        std::string fps = fp >= kMiB ? fmt("%llu MiB", (unsigned long long)(fp / kMiB))
                                     : fmt("%llu KiB", (unsigned long long)(fp / 1024));
        log("  %12s  %10.2f", fps.c_str(), best);
    }
}

// ---------------------------------------------------------------------------------------
// Peak finder (--peak): how to get the highest bandwidth out of this device, and why the
// first run after a pause can be slower. Runs entirely inside the app (no adb needed).
// ---------------------------------------------------------------------------------------

void Bench::setConfig(uint32_t wg, uint32_t threads, uint32_t line) {
    if (wg != o_.wg) {
        destroyPipelines();
        o_.wg = wg;
        createPipelines();
    }
    o_.lineBytes = line;
    vplShift_ = log2u(line / 16);
    chunkLinesBits_ = log2u(chunkBytes_ / line);
    threadCap_ = std::max<uint64_t>(threads, wg);
}

// One discarded run, then `reps` timed runs (read jobs: memory is not modified).
Bench::Stat Bench::timeJob(const Job& j0, double bytes, int reps) {
    Stat s;
    std::vector<double> g;
    Job j = j0;
    for (int i = 0; i <= reps; ++i) {
        JobResult r = execute(j);
        s.ok &= r.wgDone == r.wgExpected;
        if (i) g.push_back(bytes / (r.ms() * 1e6));
    }
    s.best = *std::max_element(g.begin(), g.end());
    s.med = median(g);
    s.runs = g;
    return s;
}

std::pair<double, double> Bench::probeVals() {
    Job s = readJob(kSeq, 0);
    Job r = readJob(kRand, readKey_ ^ 0x9E3779B9u);
    double sMs = 1e300;
    for (int i = 0; i < 3; ++i) sMs = std::min(sMs, execute(s).ms());
    double rMs = execute(r).ms();
    return {jobBytes(s) / (sMs * 1e6), jobBytes(r) / (rMs * 1e6)};
}

// Multi-threaded CPU memcpy over 256 MiB (counted as read + write). If this is also slow
// in a slow run, the memory system itself (DRAM / interconnect clock) is the limit;
// if only the GPU numbers change, it is GPU-side (GPU clock, GPU page tables / TLB).
double Bench::cpuCopyGBps() {
    const size_t bytes = 256 * kMiB;
    std::vector<uint8_t> src(bytes, 1), dst(bytes, 0);  // value-initialised: pages are mapped
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    double best = 0;
    for (int rep = 0; rep < 4; ++rep) {
        src[size_t(rep) * 4096] = uint8_t(rep + 2);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([&, t] {
                size_t b = bytes * t / nt, e = bytes * (t + 1) / nt;
                memcpy(dst.data() + b, src.data() + b, e - b);
            });
        for (auto& t : th) t.join();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (dst[size_t(rep) * 4096] != uint8_t(rep + 2)) throw std::runtime_error("CPU memcpy check failed");
        best = std::max(best, 2.0 * bytes / (ms * 1e6));
    }
    return best;
}

// Back-to-back sequential reads, reported per 1 s window: shows whether (and how fast)
// the bandwidth ramps up while the load continues.
void Bench::timeline(double ms) {
    Job j = readJob(kSeq, 0);
    const double bytes = jobBytes(j);
    auto t0 = std::chrono::steady_clock::now();
    auto sec = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    std::string line;
    double winGpuMs = 0, winBytes = 0, winEnd = 1.0, lo = 1e300, hi = 0;
    while (sec() < ms / 1000.0) {
        winGpuMs += execute(j).ms();
        winBytes += bytes;
        if (sec() >= winEnd) {
            double g = winBytes / (winGpuMs * 1e6);
            lo = std::min(lo, g);
            hi = std::max(hi, g);
            line += fmt(" %.1f", g);
            winGpuMs = winBytes = 0;
            winEnd += 1.0;
        }
    }
    log("  READ seq per second (GB/s):%s   (min %.1f, max %.1f)", line.c_str(), lo, hi);
}

// One read run with the current configuration, compared with the CPU reference.
bool Bench::checkRead(uint32_t pattern, const char* what) {
    Job j = readJob(pattern, readKey_ ^ 0x1234567u);
    JobResult r = execute(j);
    uint32_t refc[2];
    cpuReference(j, refc);
    bool ok = r.chkX == refc[0] && r.chkY == refc[1] && r.wgDone == r.wgExpected;
    verdict(ok ? "PASS" : "FAIL", fmt("%s: GPU checksum %08x:%08x %s CPU reference", what, r.chkX, r.chkY,
                                      ok ? "equals" : "DIFFERS FROM"));
    return ok;
}

void Bench::peak() {
    const auto T0 = std::chrono::steady_clock::now();
    auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count(); };
    const uint32_t wg0 = o_.wg, th0 = uint32_t(threadCap_), line0 = o_.lineBytes;
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    auto probeLog = [&](const char* when) {
        auto p = probeVals();
        log("  probe %-26s: READ seq %6.2f GB/s, READ rand %5.2f GB/s   (t=%.0f s)", when, p.first, p.second, el());
        return p;
    };

    // 1) Cold state: the very first GPU measurement of this process.
    log("");
    log("[PEAK 1/7] cold state (first measurements of this process)");
    auto cold = probeLog("cold");
    double cpuCold = cpuCopyGBps();
    log("  CPU memcpy (%u threads, R+W)   : %.2f GB/s", nt, cpuCold);

    // 2) Does continued load raise the bandwidth?
    log("");
    log("[PEAK 2/7] 8 s of continuous sequential reads");
    timeline(8000);
    auto loaded = probeLog("after 8 s load");

    // 3) Sequential read: workgroup size x invocations per dispatch.
    std::vector<uint32_t> wgs;
    for (uint32_t w : {64u, 128u, 256u, 512u, 1024u})
        if (w <= props_.limits.maxComputeWorkGroupInvocations && w <= props_.limits.maxComputeWorkGroupSize[0])
            wgs.push_back(w);
    const uint32_t ths[] = {1u << 16, 1u << 18, 1u << 20};
    log("");
    log("[PEAK 3/7] READ seq: median GB/s of 3 runs per workgroup size x invocations");
    log("  %8s %10s %10s %10s", "wg", "64K inv", "256K inv", "1M inv");
    Stat bestSeq;
    uint32_t seqWg = wg0, seqTh = th0;
    for (uint32_t w : wgs) {
        std::string line = fmt("  %8u", w);
        for (uint32_t t : ths) {
            setConfig(w, t, line0);
            Job j = readJob(kSeq, 0);
            Stat s = timeJob(j, jobBytes(j), 3);
            line += fmt(" %10.2f", s.med);
            if (s.ok && s.med > bestSeq.med) { bestSeq = s; seqWg = w; seqTh = t; }
        }
        log("%s", line.c_str());
    }
    log("  best: wg %u, %u invocations -> median %.2f GB/s, max %.2f GB/s", seqWg, seqTh, bestSeq.med, bestSeq.best);

    // 4) Random read (64 B): workgroup size x invocations, then access size.
    log("");
    log("[PEAK 4/7] READ rand %u B: median GB/s of 2 runs per workgroup size x invocations", line0);
    log("  %8s %10s %10s %10s", "wg", "64K inv", "256K inv", "1M inv");
    Stat bestRand;
    uint32_t randWg = wg0, randTh = th0;
    for (uint32_t w : wgs) {
        if (w != 64 && w != 256 && w != 1024) continue;
        std::string line = fmt("  %8u", w);
        for (uint32_t t : ths) {
            setConfig(w, t, line0);
            Job j = readJob(kRand, readKey_);
            Stat s = timeJob(j, jobBytes(j), 2);
            line += fmt(" %10.2f", s.med);
            if (s.ok && s.med > bestRand.med) { bestRand = s; randWg = w; randTh = t; }
        }
        log("%s", line.c_str());
    }
    log("  best: wg %u, %u invocations -> median %.2f GB/s (%.0f M accesses/s)", randWg, randTh, bestRand.med,
        bestRand.med * 1e3 / line0);
    log("  access size (wg %u, %u invocations):", randWg, randTh);
    log("  %8s %10s %14s", "line", "GB/s", "M accesses/s");
    for (uint32_t ln : {64u, 128u, 256u, 512u, 1024u, 4096u}) {
        setConfig(randWg, randTh, ln);
        Job j = readJob(kRand, readKey_);
        Stat s = timeJob(j, jobBytes(j), 2);
        log("  %7uB %10.2f %14.1f", ln, s.med, s.med * 1e3 / ln);
    }

    // 5) Write / copy with the best sequential configuration, and correctness of the
    //    best configurations against the CPU.
    log("");
    log("[PEAK 5/7] best configurations: write / copy, checked against the CPU");
    setConfig(seqWg, seqTh, line0);
    bool seqOk = checkRead(kSeq, fmt("READ seq (wg %u, %u inv)", seqWg, seqTh).c_str());
    rows_.push_back({"READ  seq  peak", bestSeq.runs, seqOk && bestSeq.ok});
    {
        Job j;
        j.pipe = pWrite_;
        j.linesBits = chunkLinesBits_;
        j.pattern = kSeq;
        j.numChunks = uint32_t(chunks_.size());
        const double bytes = jobBytes(j);
        std::vector<double> g;
        bool ok = true;
        for (int i = 0; i < 4; ++i) {  // new data every run, verified every run
            j.seedMix = ref::hash32(seed_ ^ (0x9E000000u + uint32_t(i)));
            JobResult r = execute(j);
            ok &= r.wgDone == r.wgExpected && verifyContent(j.seedMix, "peak write");
            curSeedMix_ = j.seedMix;
            if (i) g.push_back(bytes / (r.ms() * 1e6));
        }
        log("  WRITE seq : median %.2f GB/s, max %.2f GB/s, content %s", median(g),
            *std::max_element(g.begin(), g.end()), ok ? "verified" : "WRONG");
        verdict(ok ? "PASS" : "FAIL", "WRITE seq peak: memory content verified after every run");
        rows_.push_back({"WRITE seq  peak", g, ok});
    }
    {
        Job j;
        j.pipe = pCopy_;
        j.linesBits = chunkLinesBits_ - 1;
        j.pattern = kSeq;
        j.seedMix = curSeedMix_;
        j.dstVecOffset = 1u << (chunkLinesBits_ - 1 + vplShift_);
        j.numChunks = uint32_t(chunks_.size());
        const double bytes = 2.0 * jobBytes(j);
        std::vector<double> g;
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
            j.keyBaseB = ref::hash32(seed_ ^ 0xD5D5u) + uint32_t(i);  // the copy writes a new layout each run
            JobResult r = execute(j);
            Job v = j;
            v.pipe = pVerify_;
            v.mode = 1;
            v.unroll = 1;
            JobResult rv = execute(v);
            ok &= r.wgDone == r.wgExpected && rv.chkX == 0 && rv.wgDone == rv.wgExpected;
            if (i) g.push_back(bytes / (r.ms() * 1e6));
        }
        log("  COPY seq  : median %.2f GB/s, max %.2f GB/s (R+W), destination %s", median(g),
            *std::max_element(g.begin(), g.end()), ok ? "verified" : "WRONG");
        verdict(ok ? "PASS" : "FAIL", "COPY seq peak: destination verified after every run");
        rows_.push_back({"COPY  seq  peak", g, ok});
        fill(curSeedMix_);
    }
    setConfig(randWg, randTh, line0);
    bool randOk = checkRead(kRand, fmt("READ rand %uB (wg %u, %u inv)", line0, randWg, randTh).c_str());
    rows_.push_back({"READ  rand peak", bestRand.runs, randOk && bestRand.ok});

    // 6) Fresh allocation: free the working set and allocate it again in the same process.
    //    If the second allocation is faster, the slow state belongs to the memory the
    //    process got (page size / placement), not to the clocks.
    setConfig(wg0, th0, line0);
    log("");
    log("[PEAK 6/7] free the %llu MiB working set and allocate it again",
        (unsigned long long)(chunks_.size() * chunkBytes_ / kMiB));
    auto beforeRe = probeLog("before re-allocation");
    const size_t n = chunks_.size();
    freeChunks();
    size_t got = allocChunks(n);
    if (got < n) log("  WARNING: only %zu of %zu chunks re-allocated", got, n);
    fill(curSeedMix_);
    verdict(verifyContent(curSeedMix_, "realloc") ? "PASS" : "FAIL", "re-allocated memory content verified");
    auto afterRe = probeLog("after re-allocation");
    timeline(3000);

    // 7) Idle: does a pause bring the slow state back?
    log("");
    log("[PEAK 7/7] 15 s idle, then measure again");
    std::this_thread::sleep_for(std::chrono::seconds(15));
    auto idle = probeLog("after 15 s idle");
    double cpuEnd = cpuCopyGBps();
    log("  CPU memcpy (%u threads, R+W)   : %.2f GB/s (cold: %.2f)", nt, cpuEnd, cpuCold);

    // Diagnosis
    auto pct = [](double a, double b) { return 100.0 * (a / b - 1.0); };
    log("");
    log("==================================== PEAK RESULT ====================================");
    log("  READ seq  peak : %6.2f GB/s median (max %.2f)   --wg %u --threads %u", bestSeq.med, bestSeq.best, seqWg,
        seqTh);
    log("  READ rand peak : %6.2f GB/s median at %u B      --wg %u --threads %u", bestRand.med, line0, randWg, randTh);
    log("");
    log("  READ seq / rand over time (default config wg %u, %u invocations):", wg0, th0);
    log("    cold                 %6.2f / %5.2f", cold.first, cold.second);
    log("    after 8 s load       %6.2f / %5.2f   (%+.0f%% / %+.0f%%)", loaded.first, loaded.second,
        pct(loaded.first, cold.first), pct(loaded.second, cold.second));
    log("    before re-allocation %6.2f / %5.2f", beforeRe.first, beforeRe.second);
    log("    after re-allocation  %6.2f / %5.2f   (%+.0f%% / %+.0f%% vs before)", afterRe.first, afterRe.second,
        pct(afterRe.first, beforeRe.first), pct(afterRe.second, beforeRe.second));
    log("    after 15 s idle      %6.2f / %5.2f   (%+.0f%% / %+.0f%% vs after re-allocation)", idle.first,
        idle.second, pct(idle.first, afterRe.first), pct(idle.second, afterRe.second));
    log("    CPU memcpy           %6.2f -> %.2f GB/s (%+.0f%%)", cpuCold, cpuEnd, pct(cpuEnd, cpuCold));
    log("");
    const double kSig = 8.0;  // % treated as a real change
    bool ramp = pct(loaded.first, cold.first) > kSig || pct(loaded.second, cold.second) > kSig;
    bool realloc = pct(afterRe.first, beforeRe.first) > kSig || pct(afterRe.second, beforeRe.second) > kSig;
    bool idleDrop = pct(idle.first, afterRe.first) < -kSig || pct(idle.second, afterRe.second) < -kSig;
    bool cpuSlowCold = pct(cpuEnd, cpuCold) > kSig;
    log("  diagnosis:");
    if (ramp) log("    - bandwidth rose under continuous load: clocks ramp up (DVFS); warm up longer");
    if (realloc) log("    - a fresh allocation is faster: the slow state belongs to the memory the process got first");
    if (idleDrop) log("    - a 15 s pause lowers the bandwidth again: clocks drop when idle");
    if (cpuSlowCold) log("    - CPU memcpy was also slower at the start: the memory system itself was slower");
    if (!ramp && !realloc && !idleDrop && !cpuSlowCold)
        log("    - no first-run effect inside this process (cold = loaded = re-allocated within %.0f%%)", kSig);
    log("  total peak-finder time: %.0f s", el());
}

int Bench::summary() {
    log("");
    log("==================================== SUMMARY ====================================");
    log("  %-18s %10s %10s %10s   %s", "test", "min GB/s", "median", "max", "verified");
    double bestAll = 0;
    for (auto& r : rows_) {
        if (r.gbps.empty()) continue;
        auto mm = std::minmax_element(r.gbps.begin(), r.gbps.end());
        bestAll = std::max(bestAll, *mm.second);
        log("  %-18s %10.2f %10.2f %10.2f   %s", r.name.c_str(), *mm.first, median(r.gbps), *mm.second,
            r.verified ? "yes" : "NO");
    }
    if (o_.peakGBps > 0) {
        bool ok = bestAll <= o_.peakGBps * 1.02;
        log("");
        verdict(ok ? "PASS" : "FAIL",
                fmt("best result %.2f GB/s is %.0f%% of the given DRAM peak %.2f GB/s%s", bestAll,
                    100.0 * bestAll / o_.peakGBps, o_.peakGBps,
                    ok ? "" : " - above the physical limit: traffic did not reach DRAM"));
    }
    int fails = 0, warns = 0;
    for (auto& v : verdicts_) {
        fails += v.first == "FAIL";
        warns += v.first == "WARN";
    }
    log("");
    log("Checks: %zu total, %d FAIL, %d WARN  ->  %s", verdicts_.size(), fails, warns,
        fails ? "RESULTS ARE NOT TRUSTWORTHY" : "results verified");
    return fails ? 1 : 0;
}

int Bench::run() {
    seed_ = o_.seed;
    if (!seed_) {
        std::random_device rd;
        seed_ = rd() ^ uint32_t(std::chrono::steady_clock::now().time_since_epoch().count());
        if (!seed_) seed_ = 1;
    }
    log("membench %s - GPU-generated DRAM traffic benchmark (Vulkan compute)", kVersion);
    initVulkan();
    createResources();
    log("Seed           : %u  (reproduce with --seed %u)", seed_, seed_);
    if (!o_.peak) log("Warm-up        : %.0f ms pre-warm, then per test >= %d runs and >= %.0f ms GPU time", o_.prewarmMs,
        o_.warmup, o_.warmupMs);
    if (!o_.peak && o_.boostMs > 0 && (o_.boostCpu || o_.boostGpu))
        log("Boost          : %.0f ms %s load before the tests", o_.boostMs,
            o_.boostCpu && o_.boostGpu ? "CPU + GPU GEMM" : o_.boostCpu ? "CPU" : "GPU GEMM");
    if (o_.injectFault)
        log("*** --inject-fault: every timed kernel silently skips part of its work - checks MUST fail ***");
    readKey_ = ref::hash32(seed_ ^ 0x52454144u);

    log("");
    log("[INIT] filling the working set with seed-dependent pseudo-random data");
    fill(ref::hash32(seed_));
    verdict(verifyContent(curSeedMix_, "init") ? "PASS" : "FAIL", "initial memory content verified on the GPU");
    if (o_.peak) {
        peak();
        return summary();
    }
    prewarm();
    boost();

    std::vector<uint32_t> pats;
    if (o_.patRand) pats.push_back(kRand);
    if (o_.patSeq) pats.push_back(kSeq);

    if (o_.testRead)
        for (uint32_t p : pats) testRead(p);
    if (o_.checks) antiOptChecks();
    if (o_.testCopy)
        for (uint32_t p : pats) testCopy(p);
    if (o_.testWrite) {
        for (uint32_t p : pats) testWrite(p);
        if (o_.checks) readAfterWrite();
    }
    if (o_.sweep) sweep();
    return summary();
}

bool parseU64(const std::string& s, uint64_t& v) {
    char* end = nullptr;
    v = strtoull(s.c_str(), &end, 0);
    return end && *end == 0 && !s.empty();
}

}  // namespace

std::string usage() {
    return "usage: membench [options]\n"
           "  --size-mb N      total working set in MiB (default 1024)\n"
           "  --chunk-mb N     buffer chunk size in MiB (default auto, <= 256)\n"
           "  --line N         bytes fetched per random address: 16..4096 (default 64)\n"
           "  --reps N         measured repetitions (default 5)\n"
           "  --warmup N       minimum warm-up repetitions per test (default 2)\n"
           "  --warmup-ms N    minimum warm-up GPU time per test (default 500)\n"
           "  --prewarm-ms N   GPU load before the first test, 0 = off (default 2000)\n"
           "  --boost MODE     cpu | gpu | both | off: load right before the tests to raise\n"
           "                   the clocks; gpu = 1024^3 GEMM on the matrix units (default both)\n"
           "  --boost-ms N     duration of the boost load (default 3000)\n"
           "  --tests LIST     comma list of read,write,copy (default all)\n"
           "  --pattern P      rand | seq | both (default both)\n"
           "  --peak-gbps X    theoretical DRAM bandwidth, results above it fail\n"
           "  --sweep          working-set sweep from 64 KiB to the chunk size\n"
           "  --peak           peak finder: cold/loaded/re-allocated/idle bandwidth, best\n"
           "                   workgroup size, invocations and access size (~1.5 min)\n"
           "  --no-checks      skip the anti-optimisation checks\n"
           "  --no-cpu-ref     skip CPU reference checksums (faster)\n"
           "  --seed N         data/address seed (default random)\n"
           "  --threads N      invocations per dispatch (default 262144)\n"
           "  --wg N           workgroup size, power of 2 <= 1024 (default 256)\n"
           "  --device N       Vulkan physical device index (default 0)\n"
           "  --inject-fault   self-test: kernels skip part of their work, checks must FAIL\n";
}

bool parseArgs(const std::vector<std::string>& a, Options& o, std::string& err) {
    for (size_t i = 0; i < a.size(); ++i) {
        const std::string& k = a[i];
        auto next = [&](uint64_t& v) {
            if (i + 1 >= a.size() || !parseU64(a[i + 1], v)) {
                err = "missing/invalid value for " + k;
                return false;
            }
            ++i;
            return true;
        };
        uint64_t v = 0;
        if (k == "--help" || k == "-h") o.help = true;
        else if (k == "--size-mb") { if (!next(v)) return false; o.sizeMB = v; }
        else if (k == "--chunk-mb") { if (!next(v)) return false; o.chunkMB = uint32_t(v); }
        else if (k == "--line") { if (!next(v)) return false; o.lineBytes = uint32_t(v); }
        else if (k == "--reps") { if (!next(v)) return false; o.reps = int(std::max<uint64_t>(v, 1)); }
        else if (k == "--warmup") { if (!next(v)) return false; o.warmup = int(v); }
        else if (k == "--warmup-ms") { if (!next(v)) return false; o.warmupMs = double(v); }
        else if (k == "--prewarm-ms") { if (!next(v)) return false; o.prewarmMs = double(v); }
        else if (k == "--boost-ms") { if (!next(v)) return false; o.boostMs = double(v); }
        else if (k == "--boost") {
            if (i + 1 >= a.size()) { err = "missing value for --boost"; return false; }
            std::string m = a[++i];
            o.boostCpu = m == "cpu" || m == "both";
            o.boostGpu = m == "gpu" || m == "both";
            if (m == "off") o.boostMs = 0;
            else if (!o.boostCpu && !o.boostGpu) { err = "--boost must be cpu, gpu, both or off"; return false; }
        }
        else if (k == "--seed") { if (!next(v)) return false; o.seed = uint32_t(v); }
        else if (k == "--threads") { if (!next(v)) return false; o.threads = uint32_t(v); }
        else if (k == "--wg") { if (!next(v)) return false; o.wg = uint32_t(v); }
        else if (k == "--device") { if (!next(v)) return false; o.device = int(v); }
        else if (k == "--sweep") o.sweep = true;
        else if (k == "--peak") o.peak = true;
        else if (k == "--no-checks") o.checks = false;
        else if (k == "--no-cpu-ref") o.cpuRef = false;
        else if (k == "--inject-fault") o.injectFault = true;
        else if (k == "--peak-gbps") {
            if (i + 1 >= a.size()) { err = "missing value for --peak-gbps"; return false; }
            o.peakGBps = atof(a[++i].c_str());
        } else if (k == "--pattern") {
            if (i + 1 >= a.size()) { err = "missing value for --pattern"; return false; }
            std::string p = a[++i];
            o.patRand = p == "rand" || p == "both";
            o.patSeq = p == "seq" || p == "both";
            if (!o.patRand && !o.patSeq) { err = "--pattern must be rand, seq or both"; return false; }
        } else if (k == "--tests") {
            if (i + 1 >= a.size()) { err = "missing value for --tests"; return false; }
            std::string t = "," + a[++i] + ",";
            o.testRead = t.find(",read,") != std::string::npos;
            o.testWrite = t.find(",write,") != std::string::npos;
            o.testCopy = t.find(",copy,") != std::string::npos;
            if (!o.testRead && !o.testWrite && !o.testCopy) { err = "--tests needs read/write/copy"; return false; }
        } else {
            err = "unknown option " + k;
            return false;
        }
    }
    return true;
}

int run(const Options& o, const LogFn& log) {
    try {
        Bench b(o, log);
        return b.run();
    } catch (const std::exception& e) {
        log(std::string("ERROR: ") + e.what());
        return 2;
    }
}

}  // namespace mb
