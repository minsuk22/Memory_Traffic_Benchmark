// Random READ for the tuning sweep (--rand-tune). Same addresses, data and checksum as
// read.comp, but the kernel shape is a specialization constant:
//   UNROLL = random lines each invocation has in flight per loop iteration (1..16)
//   VPL    = consecutive uvec4 per line one invocation loads (1, 2, 4: 16 / 32 / 64 B),
//            so 16 << vplShift B lines are fetched by (lines / VPL) neighbouring lanes
// UNROLL * VPL <= 16 (checked by the host). pc.dstVecOffset is the window base (uvec4
// index inside the bound chunk): one dispatch reads one window.
//
// The checksum is order independent, and every logical index g is loaded exactly once for
// any UNROLL / VPL / thread count, so all shapes must give the CPU reference checksum.
//
// Defines: TEXEL = load through a uniform texel buffer (texture path) instead of the SSBO.

layout(constant_id = 1) const uint UNROLL = 4u;
layout(constant_id = 2) const uint VPL = 1u;
// ROLL = 1: rolling windows. One dispatch covers many consecutive windows; workgroup g
// works on window g / pc.keyB.x (keyB.x = workgroups per window), so with the GPU starting
// workgroups roughly in ID order only a few windows are active at any time, without a
// barrier between windows. keyA.x = permutation key base (keys derived per window here,
// exactly like deriveKey() on the host), keyB.y = global index of the first window,
// keyB.z = uvec4 per window; pc.numThreads = invocations per window.
layout(constant_id = 3) const uint ROLL = 0u;
// SPREAD_SHIFT = s > 0: only one line per 2^s line slots is read (line i at slot i << s), so
// no two lines read share a 2^s-line block. Checks whether small windows are fast only
// because the other half of a larger cache line is hit later (that would not be random).
layout(constant_id = 4) const uint SPREAD_SHIFT = 0u;
// WRITE_PCT (MIX_RW kernel only): a line is written instead of read if
// hash32(global line ^ seedMix ^ salt) % 100 < WRITE_PCT. The choice depends on the physical
// line only, so every kernel shape / window writes the same lines and reads all the others
// (whose content never changes). A written line gets its canonical value XOR pc.mode
// (the host's write tag): tag != 0 makes every written 16 B word differ from the fill, so
// the host can count them afterwards (verify.comp), tag 0 restores the fill.
layout(constant_id = 5) const uint WRITE_PCT = 0u;
const uint NL = UNROLL * VPL;

#include "common.glsl"

uint spreadIndex(uint vi) {
    if (SPREAD_SHIFT == 0u) return vi;
    return ((vi >> pc.vplShift) << (pc.vplShift + SPREAD_SHIFT)) | (vi & ((1u << pc.vplShift) - 1u));
}

#ifdef TEXEL
layout(set = 0, binding = 2) uniform usamplerBuffer tex;
#define LOAD(i) texelFetch(tex, int(i))
#elif defined(MIX_RW)
layout(std430, set = 0, binding = 0) buffer Data { uvec4 data[]; };
#define LOAD(i) data[i]
#else
layout(std430, set = 0, binding = 0) readonly buffer Data { uvec4 data[]; };
#define LOAD(i) data[i]
#endif

#ifdef MIX_RW
uint gvBase;  // global uvec4 index of uvec4 0 of the bound chunk
bool isWrite(uint vi) {  // vi = uvec4 index inside the chunk
    return hash32(((gvBase + vi) >> pc.vplShift) ^ pc.seedMix ^ 0x6A09E667u) % 100u < WRITE_PCT;
}
uvec4 written(uint vi) {
    uint w = (gvBase + vi) * 4u;
    return uvec4(dataValue(w), dataValue(w + 1u), dataValue(w + 2u), dataValue(w + 3u)) ^ uvec4(pc.mode);
}
// Line slot k / VPL, uvec4 k % VPL of it: written (and left out of the checksum) or read.
#define LD(k)                                                                  \
    uvec4 v##k = uvec4(0u);                                                    \
    uint g##k = 0u;                                                            \
    bool r##k = false;                                                         \
    if (k < NL) {                                                              \
        g##k = ((it + k / VPL) * T + tid) * VPL + k % VPL;                     \
        uint i##k = base + spreadIndex(vecIndex(g##k, key));                   \
        r##k = !isWrite(i##k);                                                 \
        if (r##k) v##k = LOAD(i##k);                                           \
        else data[i##k] = written(i##k);                                       \
    }
#define MIX(k) if (k < NL && r##k) acc = mixChecksum(acc, v##k, lbase + g##k);
#else
// Load k of one iteration: line slot k / VPL, uvec4 k % VPL of it. Branches on k < NL fold
// away after specialization, so the loads are plain registers (no private array).
#define LD(k)                                                                  \
    uvec4 v##k = uvec4(0u);                                                    \
    uint g##k = 0u;                                                            \
    if (k < NL) {                                                              \
        g##k = ((it + k / VPL) * T + tid) * VPL + k % VPL;                     \
        v##k = LOAD(base + spreadIndex(vecIndex(g##k, key)));                  \
    }
#define MIX(k) if (k < NL) acc = mixChecksum(acc, v##k, lbase + g##k);
#endif

// Host deriveKey(base, w): per-window permutation key.
uvec4 windowKey(uint b, uint w) {
    return uvec4(hash32(b + 0x9E3779B9u * (w * 4u + 1u)), hash32(b + 0x9E3779B9u * (w * 4u + 2u)),
                 hash32(b + 0x9E3779B9u * (w * 4u + 3u)), hash32(b + 0x9E3779B9u * (w * 4u + 4u)));
}

void main() {
    uint tid   = gl_GlobalInvocationID.x;
    uint T     = pc.numThreads;
    uint base  = pc.dstVecOffset;
    uint lbase = pc.logicalBase;
    uvec4 key  = pc.keyA;
    if (ROLL != 0u) {
        uint wl = gl_WorkGroupID.x / pc.keyB.x;  // window inside this dispatch
        tid   = (gl_WorkGroupID.x % pc.keyB.x) * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
        base  = pc.dstVecOffset + wl * pc.keyB.w;  // keyB.w = uvec4 of memory per window
        lbase = pc.logicalBase + wl * pc.keyB.z;
        key   = windowKey(pc.keyA.x, pc.keyB.y + wl);
    }
#ifdef MIX_RW
    gvBase = (pc.chunkWordBase >> 2) - pc.dstVecOffset;  // chunkWordBase = chunk start + window offset
#endif
    uvec2 acc = uvec2(0u);

    for (uint it = 0u; it < pc.itersPerThread; it += UNROLL) {
        LD(0) LD(1) LD(2) LD(3) LD(4) LD(5) LD(6) LD(7)
        LD(8) LD(9) LD(10) LD(11) LD(12) LD(13) LD(14) LD(15)
        MIX(0) MIX(1) MIX(2) MIX(3) MIX(4) MIX(5) MIX(6) MIX(7)
        MIX(8) MIX(9) MIX(10) MIX(11) MIX(12) MIX(13) MIX(14) MIX(15)
    }

    publish(acc);
}
