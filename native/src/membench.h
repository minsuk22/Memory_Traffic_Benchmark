// GPU-driven DRAM traffic benchmark (Vulkan compute).
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mb {

struct Options {
    uint64_t sizeMB    = 1024;  // total working set
    uint32_t chunkMB   = 0;     // 0 = auto (<= 256 MiB, <= device limits)
    uint32_t lineBytes = 64;    // bytes fetched per random address (16..4096, power of 2)
    int      reps      = 5;
    int      warmup    = 2;
    bool     testRead  = true;
    bool     testWrite = true;
    bool     testCopy  = true;
    bool     patRand   = true;
    bool     patSeq    = true;
    bool     checks    = true;  // anti-optimisation checks
    bool     cpuRef    = true;  // CPU reference checksum
    bool     sweep     = false; // working-set sweep (cache -> DRAM)
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
