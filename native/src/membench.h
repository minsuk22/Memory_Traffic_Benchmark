// GPU-driven DRAM traffic benchmark (Vulkan compute).
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mb {

constexpr const char* kVersion = "1.6";

struct Options {
    uint64_t sizeMB    = 1024;  // total working set
    uint32_t chunkMB   = 0;     // 0 = auto (<= 256 MiB, <= device limits)
    uint32_t lineBytes = 64;    // bytes fetched per random address (16..4096, power of 2)
    int      reps      = 5;
    int      warmup    = 2;     // minimum warm-up runs per test
    double   warmupMs  = 500;   // ... and minimum warm-up GPU time per test
    double   prewarmMs = 2000;  // GPU load before the first test (clock ramp-up)
    double   boostMs   = 3000;  // CPU + GPU (1024^3 GEMM) load right before the tests, 0 = off
    bool     boostCpu  = true;
    bool     boostGpu  = true;
    bool     testRead  = true;
    bool     testWrite = true;
    bool     testCopy  = true;
    bool     patRand   = true;
    bool     patSeq    = true;
    bool     checks    = true;  // anti-optimisation checks
    bool     cpuRef    = true;  // CPU reference checksum
    bool     sweep     = false; // working-set sweep (cache -> DRAM)
    bool     peak      = false; // peak finder: timeline, config search, re-allocation, idle tests
    bool     cpu       = false; // CPU random read peak (same addresses/data/checksum as the GPU)
    bool     dvfs      = false; // random read vs memory clock governors (background traffic, slices, footprint)
    double   peakGBps  = 0;     // theoretical DRAM peak, 0 = unknown
    uint32_t seed      = 0;     // 0 = random per run
    uint32_t threads   = 0;     // invocations per dispatch, 0 = auto
    uint32_t wg        = 256;   // workgroup size (power of 2)
    int      device    = 0;
    bool     injectFault = false;  // self-test: kernels silently skip part of their work
    bool     help      = false;
};

using LogFn = std::function<void(const std::string& line)>;

bool parseArgs(const std::vector<std::string>& args, Options& o, std::string& err);
std::string usage();

// Returns 0 if every check passed, 1 if any check failed, 2 on a setup/runtime error.
int run(const Options& o, const LogFn& log);

}  // namespace mb
