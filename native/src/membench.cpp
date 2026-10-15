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
    void fill(uint32_t seedMix);
    bool verifyContent(uint32_t seedMix, const char* what);
    void testRead(uint32_t pattern);
    void antiOptChecks();
    void testCopy(uint32_t pattern);
    void testWrite(uint32_t pattern);
    void readAfterWrite();
    void sweep();
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
    app.apiVersion = VK_API_VERSION_1_1;
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

    VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &m3;
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

    // Optional: driver statistics of the compiled pipelines (instruction counts etc.).
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, exts.data());
    bool hasStatsExt = false;
    for (auto& e : exts)
        if (!strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) hasStatsExt = true;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pef{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    if (hasStatsExt) {
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &pef;
        vkGetPhysicalDeviceFeatures2(phys_, &f2);
        statsExt_ = pef.pipelineExecutableInfo == VK_TRUE;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* extNames[] = {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR enable{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    enable.pipelineExecutableInfo = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (statsExt_) {
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = extNames;
        dci.pNext = &enable;
    }
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
    int dataType = -1;
    for (uint64_t c = 0; c < nChunks; ++c) {
        Chunk ch;
        bci.size = chunk;
        VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &ch.buf));
        vkGetBufferMemoryRequirements(dev_, ch.buf, &mr);
        const VkMemoryPropertyFlags never =
            VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT;
        // Prefer plain device-local memory (no CPU-cached/snooped types).
        dataType = pickMemType(memProps_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               never | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (dataType < 0) dataType = pickMemType(memProps_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, never);
        if (dataType < 0) dataType = pickMemType(memProps_, mr.memoryTypeBits, 0, never);
        if (dataType < 0) throw std::runtime_error("no usable memory type");
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = (uint32_t)dataType;
        VkResult r = vkAllocateMemory(dev_, &mai, nullptr, &ch.mem);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(dev_, ch.buf, nullptr);
            if (chunks_.empty()) VK_CHECK(r);
            log("WARNING: allocation of chunk %llu failed (VkResult %d); continuing with %zu chunk(s)",
                (unsigned long long)c, (int)r, chunks_.size());
            break;
        }
        VK_CHECK(vkBindBufferMemory(dev_, ch.buf, ch.mem, 0));
        chunks_.push_back(ch);
    }

    for (auto& ch : chunks_) {
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
    }

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

    // --- pipelines --------------------------------------------------------------------
    pRead_     = makePipeline(kReadSpv, sizeof kReadSpv, 1);
    pReadDead_ = makePipeline(kReadSpv, sizeof kReadSpv, 0);
    pWrite_    = makePipeline(kWriteSpv, sizeof kWriteSpv, 1);
    pCopy_     = makePipeline(kCopySpv, sizeof kCopySpv, 1);
    pVerify_   = makePipeline(kVerifySpv, sizeof kVerifySpv, 1);

    log("Data memory    : type %d [%s]", dataType,
        memFlagsStr(memProps_.memoryTypes[dataType].propertyFlags).c_str());
    log("Working set    : %llu MiB = %zu chunk(s) x %llu MiB  (requested %llu MiB)",
        (unsigned long long)(chunks_.size() * chunkBytes_ / kMiB), chunks_.size(),
        (unsigned long long)(chunkBytes_ / kMiB), (unsigned long long)o_.sizeMB);
    log("Access         : %u B per random address (%u lanes x 16 B), %llu lines per chunk",
        o_.lineBytes, 1u << vplShift_, (unsigned long long)(1ull << chunkLinesBits_));
    log("Dispatch       : workgroup %u, up to %llu invocations, 4 independent loads per iteration",
        o_.wg, (unsigned long long)threadCap_);
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
    log("Warm-up        : %.0f ms pre-warm, then per test >= %d runs and >= %.0f ms GPU time", o_.prewarmMs,
        o_.warmup, o_.warmupMs);
    if (o_.injectFault)
        log("*** --inject-fault: every timed kernel silently skips part of its work - checks MUST fail ***");
    readKey_ = ref::hash32(seed_ ^ 0x52454144u);

    log("");
    log("[INIT] filling the working set with seed-dependent pseudo-random data");
    fill(ref::hash32(seed_));
    verdict(verifyContent(curSeedMix_, "init") ? "PASS" : "FAIL", "initial memory content verified on the GPU");
    prewarm();

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
           "  --tests LIST     comma list of read,write,copy (default all)\n"
           "  --pattern P      rand | seq | both (default both)\n"
           "  --peak-gbps X    theoretical DRAM bandwidth, results above it fail\n"
           "  --sweep          working-set sweep from 64 KiB to the chunk size\n"
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
        else if (k == "--seed") { if (!next(v)) return false; o.seed = uint32_t(v); }
        else if (k == "--threads") { if (!next(v)) return false; o.threads = uint32_t(v); }
        else if (k == "--wg") { if (!next(v)) return false; o.wg = uint32_t(v); }
        else if (k == "--device") { if (!next(v)) return false; o.device = int(v); }
        else if (k == "--sweep") o.sweep = true;
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
