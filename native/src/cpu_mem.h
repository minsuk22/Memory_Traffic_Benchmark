// CPU random-read traffic with exactly the addresses, data and checksum of the GPU read
// kernel (read.comp / reference.h). The CPU result therefore must equal the checksum
// formula and the GPU result for the same job: every line was really loaded.
#pragma once
#include <array>
#include <cstdint>
#include <string>
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

}  // namespace cpu
}  // namespace mb
