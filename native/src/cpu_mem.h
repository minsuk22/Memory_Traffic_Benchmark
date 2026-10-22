// CPU random-read traffic with exactly the addresses, data and checksum of the GPU read
// kernel (read.comp / reference.h). The CPU result therefore must equal the checksum
// formula and the GPU result for the same job: every line was really loaded.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mb {
namespace cpu {

// Host memory laid out like the GPU chunks (chunk c starts at word c * chunkWords).
class Buffer {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { release(); }
    // huge: ask for transparent huge pages (MADV_HUGEPAGE) where the kernel offers them.
    void allocate(uint64_t bytes, bool huge);
    void release();
    void fill(uint32_t seedMix, unsigned threads);  // word w = hash32(w ^ seedMix), like the GPU
    const uint32_t* words() const { return w_; }

private:
    uint32_t* w_ = nullptr;
    void* base_ = nullptr;
    uint64_t bytes_ = 0, mapped_ = 0;
};

// Same description as a GPU random read job.
struct ReadCfg {
    uint32_t linesBits = 0;  // permutation domain per chunk = 2^linesBits lines
    uint32_t vplShift = 0;   // line = 16 << vplShift bytes
    uint32_t passes = 1;
    uint32_t seedMix = 0;
    uint32_t chunkWords = 0;
    std::vector<std::array<uint32_t, 4>> keys;  // permutation key per chunk
    uint64_t linesPerChunk = 0;  // logical lines read per chunk, 0 = all passes
    unsigned threads = 1;
    uint32_t prefetch = 0;   // software prefetch distance in lines (power of two <= 256), 0 = off
    bool dependent = false;  // next address depends on the previous load (latency test)
    uint32_t spread = 0;     // line i lives at line slot i << spread (cache-line reuse check)
};

struct ReadResult {
    double ms = 0;
    uint32_t chk[2] = {0, 0};
    uint64_t lines = 0;  // lines read
};

ReadResult randRead(const Buffer& b, const ReadCfg& c);
// Checksum the read must produce, computed from the formula (no memory reads).
void expectedChecksum(const ReadCfg& c, uint32_t out[2]);
// Transparent-huge-page mode and AnonHugePages of this process (best effort).
std::string hugePageInfo();

// Background sequential reads on `threads` cores over a buffer (>> caches), so the memory
// clock governors see high DRAM bandwidth while something else is measured.
class StreamLoad {
public:
    StreamLoad() = default;
    StreamLoad(const StreamLoad&) = delete;
    StreamLoad& operator=(const StreamLoad&) = delete;
    ~StreamLoad() { stop(); }
    void start(const uint32_t* words, uint64_t bytes, unsigned threads);
    void stop();
    void mark();                 // starts a measurement window
    double gbpsSinceMark() const;

private:
    std::vector<std::thread> th_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> bytes_{0};
    uint64_t markBytes_ = 0;
    std::chrono::steady_clock::time_point markT_;
};

// Clock files (devfreq cur_freq, kgsl gpuclk) the app is allowed to read. Usually none on
// production phones (SELinux), in which case the sampler reports nothing.
struct ClockFile {
    std::string name, path;
};
std::vector<ClockFile> findClockFiles();

// Samples the clock files every 20 ms between start() and stop().
class ClockSampler {
public:
    explicit ClockSampler(const std::vector<ClockFile>& f) : files_(f) {}
    ~ClockSampler() { stop(); }
    void start();
    std::string stop();  // "name min-max MHz, ..." ("" without readable files)

private:
    std::vector<ClockFile> files_;
    std::vector<std::pair<double, double>> range_;
    std::thread th_;
    std::atomic<bool> stop_{false};
};

}  // namespace cpu
}  // namespace mb
