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
const uint NL = UNROLL * VPL;

#include "common.glsl"

#ifdef TEXEL
layout(set = 0, binding = 2) uniform usamplerBuffer tex;
#define LOAD(i) texelFetch(tex, int(i))
#else
layout(std430, set = 0, binding = 0) readonly buffer Data { uvec4 data[]; };
#define LOAD(i) data[i]
#endif

// Load k of one iteration: line slot k / VPL, uvec4 k % VPL of it. Branches on k < NL fold
// away after specialization, so the loads are plain registers (no private array).
#define LD(k)                                                                  \
    uvec4 v##k = uvec4(0u);                                                    \
    uint g##k = 0u;                                                            \
    if (k < NL) {                                                              \
        g##k = ((it + k / VPL) * T + tid) * VPL + k % VPL;                     \
        v##k = LOAD(base + vecIndex(g##k, pc.keyA));                           \
    }
#define MIX(k) if (k < NL) acc = mixChecksum(acc, v##k, pc.logicalBase + g##k);

void main() {
    uint tid  = gl_GlobalInvocationID.x;
    uint T    = pc.numThreads;
    uint base = pc.dstVecOffset;
    uvec2 acc = uvec2(0u);

    for (uint it = 0u; it < pc.itersPerThread; it += UNROLL) {
        LD(0) LD(1) LD(2) LD(3) LD(4) LD(5) LD(6) LD(7)
        LD(8) LD(9) LD(10) LD(11) LD(12) LD(13) LD(14) LD(15)
        MIX(0) MIX(1) MIX(2) MIX(3) MIX(4) MIX(5) MIX(6) MIX(7)
        MIX(8) MIX(9) MIX(10) MIX(11) MIX(12) MIX(13) MIX(14) MIX(15)
    }

    publish(acc);
}
