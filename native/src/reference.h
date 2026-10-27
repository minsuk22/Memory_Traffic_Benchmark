// Bit-exact CPU mirror of native/shaders/common.glsl. Used to compute the checksum the
// GPU *must* produce if (and only if) it really loaded every address it was asked to.
#pragma once
#include <cstdint>

namespace ref {

struct Params {
    uint32_t linesMask, linesBits, vplShift, pattern, seedMix;
    uint32_t keyA[4];
    uint32_t spreadShift;  // read_tune SPREAD_SHIFT: line i lives at line slot i << spreadShift
    uint32_t writePct;     // read_tune_mix WRITE_PCT: written lines are left out of the checksum
};

inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline uint32_t dataValue(uint32_t word, uint32_t seedMix) { return hash32(word ^ seedMix); }

inline uint32_t permuteLine(const Params& p, uint32_t x, const uint32_t k[4], uint32_t pass) {
    if (p.linesBits == 0u) return 0u;
    const uint32_t m  = p.linesMask;
    uint32_t sh = (p.linesBits + 1u) >> 1;
    if (sh < 1u) sh = 1u;
    x = ((x ^ (k[0] ^ (pass * 0x9E3779B9u))) * 0x9E3779B1u) & m; x ^= x >> sh;
    x = ((x ^ k[1]) * 0x85EBCA6Bu) & m; x ^= x >> sh;
    x = ((x ^ k[2]) * 0xC2B2AE35u) & m; x ^= x >> sh;
    x = ((x ^ k[3]) * 0x27D4EB2Fu) & m; x ^= x >> sh;
    return x;
}

inline uint32_t vecIndex(const Params& p, uint32_t g, const uint32_t k[4]) {
    uint32_t lineL = g >> p.vplShift;
    uint32_t sub   = g & ((1u << p.vplShift) - 1u);
    uint32_t line  = lineL & p.linesMask;
    if (p.pattern != 0u) line = permuteLine(p, line, k, lineL >> p.linesBits);
    return (line << p.vplShift) | sub;
}

// read_tune_mix: is global line `line` written instead of read?
inline bool isWriteLine(uint32_t line, uint32_t seedMix, uint32_t writePct) {
    return writePct && hash32(line ^ seedMix ^ 0x6A09E667u) % 100u < writePct;
}

inline uint32_t rotl(uint32_t x, uint32_t r) { return (x << r) | (x >> (32u - r)); }

inline void mixChecksum(uint32_t acc[2], const uint32_t v[4], uint32_t G) {
    acc[0] += (v[0] ^ G) + rotl(v[1], 7u) + rotl(v[2], 13u) + rotl(v[3], 19u);
    acc[1] += (rotl(v[0] ^ v[3], 5u) * (G | 1u)) + (v[1] ^ v[2]);
}

// Checksum that read.comp produces for logical indices [gBegin, gEnd) of one chunk.
inline void readChecksum(const Params& p, uint32_t chunkWordBase, uint32_t logicalBase,
                         uint32_t gBegin, uint32_t gEnd, uint32_t acc[2]) {
    for (uint32_t g = gBegin; g < gEnd; ++g) {
        uint32_t vi = vecIndex(p, g, p.keyA);
        if (p.spreadShift)
            vi = ((vi >> p.vplShift) << (p.vplShift + p.spreadShift)) | (vi & ((1u << p.vplShift) - 1u));
        uint32_t w  = chunkWordBase + vi * 4u;
        if (isWriteLine((w >> 2) >> p.vplShift, p.seedMix, p.writePct)) continue;
        uint32_t v[4] = {dataValue(w, p.seedMix), dataValue(w + 1u, p.seedMix),
                         dataValue(w + 2u, p.seedMix), dataValue(w + 3u, p.seedMix)};
        mixChecksum(acc, v, logicalBase + g);
    }
}

}  // namespace ref
