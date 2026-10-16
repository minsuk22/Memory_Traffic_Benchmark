#include "cpu_mem.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <thread>

#include "reference.h"

#ifdef _WIN32
#include <malloc.h>
#else
#include <dirent.h>
#include <sys/mman.h>
#endif

namespace mb {
namespace cpu {
namespace {

constexpr uint64_t kHuge = 2ull << 20;

// Read once per run: the compiler cannot know it is 0, so `x & gZero` keeps a real data
// dependency in the latency test without changing any address.
volatile uint32_t gZero = 0;
std::atomic<uint64_t> gSink{0};

inline void mix(uint32_t& a0, uint32_t& a1, const uint32_t* v, uint32_t G) {
    a0 += (v[0] ^ G) + ref::rotl(v[1], 7u) + ref::rotl(v[2], 13u) + ref::rotl(v[3], 19u);
    a1 += (ref::rotl(v[0] ^ v[3], 5u) * (G | 1u)) + (v[1] ^ v[2]);
}

uint64_t logicalLines(const ReadCfg& c) {
    return c.linesPerChunk ? c.linesPerChunk : uint64_t(c.passes) << c.linesBits;
}

// Logical lines [lb, le) of one chunk; same address and checksum order as read.comp.
template <bool PF, bool DEP>
void readRange(const uint32_t* words, const ReadCfg& c, uint32_t chunk, uint32_t lb, uint32_t le, uint32_t zero,
               uint32_t acc[2]) {
    const ref::Params p{(1u << c.linesBits) - 1u, c.linesBits, c.vplShift, 1u, c.seedMix, {}};
    const uint32_t* k = c.keys[chunk].data();
    const uint32_t vpl = 1u << c.vplShift;
    const uint32_t lineWords = vpl * 4u;
    const uint32_t* base = words + uint64_t(chunk) * c.chunkWords;
    const uint32_t logicalBase = uint32_t(uint64_t(chunk) * (uint64_t(c.passes) << (c.linesBits + c.vplShift)));
    auto lineOf = [&](uint32_t lineL) { return ref::permuteLine(p, lineL & p.linesMask, k, lineL >> c.linesBits); };
    auto prefetchLine = [&](uint32_t line) {
        const char* a = reinterpret_cast<const char*>(base + uint64_t(line) * lineWords);
        for (uint32_t o = 0; o < lineWords * 4u; o += 64u) __builtin_prefetch(a + o);
    };

    uint32_t ring[256];
    const uint32_t D = PF ? c.prefetch : 0u, mask = D - 1u;
    if (PF)
        for (uint32_t i = 0; i < D && lb + i < le; ++i) {
            ring[i] = lineOf(lb + i);
            prefetchLine(ring[i]);
        }
    uint32_t a0 = acc[0], a1 = acc[1], dep = 0;
    for (uint32_t lineL = lb; lineL < le; ++lineL) {
        uint32_t line;
        if (PF) {
            const uint32_t s = (lineL - lb) & mask;
            line = ring[s];
            if (le - lineL > D) {
                const uint32_t n = lineOf(lineL + D);
                ring[s] = n;
                prefetchLine(n);
            }
        } else if (DEP) {
            line = lineOf(lineL ^ (dep & zero));
        } else {
            line = lineOf(lineL);
        }
        const uint32_t* v = base + uint64_t(line) * lineWords;
        const uint32_t g = logicalBase + (lineL << c.vplShift);
        for (uint32_t s = 0; s < vpl; ++s) mix(a0, a1, v + 4u * s, g + s);
        if (DEP) dep = v[0];
    }
    acc[0] = a0;
    acc[1] = a1;
}

// Splits all (chunk, logical line) pairs evenly over the threads.
template <typename Fn>
void forEachRange(const ReadCfg& c, unsigned t, unsigned nt, Fn&& fn) {
    const uint64_t L = logicalLines(c), total = L * c.keys.size();
    uint64_t b = total * t / nt, e = total * (t + 1) / nt;
    while (b < e) {
        const uint64_t chunk = b / L, lb = b % L;
        const uint64_t le = std::min<uint64_t>(L, lb + (e - b));
        fn(uint32_t(chunk), uint32_t(lb), uint32_t(le));
        b += le - lb;
    }
}

}  // namespace

void Buffer::allocate(uint64_t bytes, bool huge) {
    release();
    bytes_ = bytes;
#ifdef _WIN32
    (void)huge;
    base_ = _aligned_malloc(bytes, kHuge);
    if (!base_) throw std::bad_alloc();
    w_ = static_cast<uint32_t*>(base_);
#else
    mapped_ = bytes + kHuge;  // room to align the start to 2 MiB
    base_ = mmap(nullptr, mapped_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        throw std::bad_alloc();
    }
    uintptr_t a = (reinterpret_cast<uintptr_t>(base_) + kHuge - 1) & ~uintptr_t(kHuge - 1);
    w_ = reinterpret_cast<uint32_t*>(a);
#ifdef MADV_HUGEPAGE
    madvise(w_, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
#else
    (void)huge;
#endif
#endif
}

void Buffer::release() {
    if (!base_) return;
#ifdef _WIN32
    (void)mapped_;
    _aligned_free(base_);
#else
    munmap(base_, mapped_);
#endif
    base_ = nullptr;
    w_ = nullptr;
}

void Buffer::fill(uint32_t seedMix, unsigned threads) {
    const uint64_t n = bytes_ / 4;
    std::vector<std::thread> th;
    for (unsigned t = 0; t < threads; ++t)
        th.emplace_back([=] {
            for (uint64_t w = n * t / threads, e = n * (t + 1) / threads; w < e; ++w)
                w_[w] = ref::dataValue(uint32_t(w), seedMix);
        });
    for (auto& t : th) t.join();
}

ReadResult randRead(const Buffer& b, const ReadCfg& c) {
    const unsigned nt = std::max(1u, c.threads);
    std::vector<std::array<uint32_t, 2>> part(nt, {0u, 0u});
    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            const uint32_t zero = gZero;
            ++ready;
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            forEachRange(c, t, nt, [&](uint32_t chunk, uint32_t lb, uint32_t le) {
                if (c.dependent) readRange<false, true>(b.words(), c, chunk, lb, le, zero, part[t].data());
                else if (c.prefetch) readRange<true, false>(b.words(), c, chunk, lb, le, zero, part[t].data());
                else readRange<false, false>(b.words(), c, chunk, lb, le, zero, part[t].data());
            });
        });
    while (ready.load() < nt) std::this_thread::yield();  // thread start-up is not timed
    auto t0 = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& t : th) t.join();
    ReadResult r;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (auto& p : part) {
        r.chk[0] += p[0];
        r.chk[1] += p[1];
    }
    r.lines = logicalLines(c) * c.keys.size();
    return r;
}

void expectedChecksum(const ReadCfg& c, uint32_t out[2]) {
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::array<uint32_t, 2>> part(nt, {0u, 0u});
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            forEachRange(c, t, nt, [&](uint32_t chunk, uint32_t lb, uint32_t le) {
                ref::Params p{(1u << c.linesBits) - 1u, c.linesBits, c.vplShift, 1u, c.seedMix, {}};
                std::memcpy(p.keyA, c.keys[chunk].data(), sizeof p.keyA);
                const uint32_t logicalBase =
                    uint32_t(uint64_t(chunk) * (uint64_t(c.passes) << (c.linesBits + c.vplShift)));
                ref::readChecksum(p, uint32_t(uint64_t(chunk) * c.chunkWords), logicalBase, lb << c.vplShift,
                                  le << c.vplShift, part[t].data());
            });
        });
    for (auto& t : th) t.join();
    out[0] = out[1] = 0;
    for (auto& p : part) {
        out[0] += p[0];
        out[1] += p[1];
    }
}

std::string hugePageInfo() {
    std::string mode = "unknown";
    {
        std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
        std::string s;
        if (std::getline(f, s) && !s.empty()) mode = s;
    }
    std::string anon = "?";
    {
        std::ifstream f("/proc/self/smaps_rollup");
        for (std::string s; std::getline(f, s);)
            if (s.rfind("AnonHugePages:", 0) == 0) {
                anon = s.substr(14);
                anon.erase(0, anon.find_first_not_of(' '));
            }
    }
    return "THP mode: " + mode + ", AnonHugePages of this process: " + anon;
}

void StreamLoad::start(const uint32_t* words, uint64_t bytes, unsigned threads) {
    stop();
    stop_ = false;
    bytes_ = 0;
    const uint64_t block = 1ull << 20;
    const uint64_t blocks = bytes / block;
    for (unsigned t = 0; t < threads; ++t)
        th_.emplace_back([=] {
            const uint64_t* w = reinterpret_cast<const uint64_t*>(words);
            const uint64_t b0 = blocks * t / threads, b1 = std::max(b0 + 1, blocks * (t + 1) / threads);
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (uint64_t b = b0; !stop_.load(std::memory_order_relaxed); b = b + 1 < b1 ? b + 1 : b0) {
                const uint64_t* p = w + b * (block / 8);
                for (uint64_t i = 0; i < block / 8; i += 4) {
                    s0 += p[i];
                    s1 ^= p[i + 1];
                    s2 += p[i + 2];
                    s3 ^= p[i + 3];
                }
                bytes_.fetch_add(block, std::memory_order_relaxed);
            }
            gSink.fetch_xor(s0 + s1 + s2 + s3, std::memory_order_relaxed);  // keeps the loads alive
        });
    mark();
}

void StreamLoad::stop() {
    stop_ = true;
    for (auto& t : th_) t.join();
    th_.clear();
}

void StreamLoad::mark() {
    markBytes_ = bytes_.load();
    markT_ = std::chrono::steady_clock::now();
}

double StreamLoad::gbpsSinceMark() const {
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - markT_).count();
    return s > 0 ? double(bytes_.load() - markBytes_) / (s * 1e9) : 0.0;
}

namespace {
bool readNumber(const std::string& path, double& v) {
    std::ifstream f(path);
    return bool(f >> v);
}
}  // namespace

std::vector<ClockFile> findClockFiles() {
    std::vector<ClockFile> out;
    double v;
#ifndef _WIN32
    const ClockFile kgsl[] = {{"gpuclk", "/sys/class/kgsl/kgsl-3d0/gpuclk"},
                              {"kgsl devfreq", "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq"}};
    for (const ClockFile& f : kgsl)
        if (readNumber(f.path, v)) out.push_back(f);
    if (DIR* d = opendir("/sys/class/devfreq")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            std::string path = std::string("/sys/class/devfreq/") + e->d_name + "/cur_freq";
            if (readNumber(path, v)) out.push_back({e->d_name, path});
        }
        closedir(d);
    }
#else
    (void)v;
#endif
    return out;
}

void ClockSampler::start() {
    stop();
    stop_ = false;
    range_.assign(files_.size(), {1e300, 0.0});
    if (files_.empty()) return;
    th_ = std::thread([this] {
        while (!stop_.load()) {
            for (size_t i = 0; i < files_.size(); ++i) {
                double v;
                if (!readNumber(files_[i].path, v)) continue;
                range_[i].first = std::min(range_[i].first, v);
                range_[i].second = std::max(range_[i].second, v);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
}

std::string ClockSampler::stop() {
    stop_ = true;
    if (th_.joinable()) th_.join();
    std::string s;
    for (size_t i = 0; i < range_.size(); ++i) {
        if (range_[i].second <= 0) continue;
        // Hz values are shown in MHz; bandwidth-vote devices (MB/s, kHz) are shown as read.
        const bool hz = range_[i].second >= 1e7;
        const double div = hz ? 1e6 : 1.0;
        char buf[160];
        snprintf(buf, sizeof buf, "%s%s %.0f-%.0f%s", s.empty() ? "" : ", ", files_[i].name.c_str(),
                 range_[i].first / div, range_[i].second / div, hz ? " MHz" : "");
        s += buf;
    }
    range_.clear();
    return s;
}

}  // namespace cpu
}  // namespace mb
