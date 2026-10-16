// Random READ feeding the matrix units (VK_KHR_cooperative_matrix): the same loads and
// checksum as read.comp, and the loaded data becomes the A operand of coopMatMulAdd.
// Each invocation loads 4 x 16 B per iteration (32 fp16 values); the subgroup stores them
// to shared memory and multiplies every lM x lK tile with a fixed B tile pc.mode times
// (pc.mode = 0: same stores and barriers, no matrix work). 16 flop per loaded byte per rep.
// One workgroup = one subgroup.
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(constant_id = 2) const uint lM = 16;
layout(constant_id = 3) const uint lN = 16;
layout(constant_id = 4) const uint lK = 16;

#include "common.glsl"

layout(std430, set = 0, binding = 0) readonly buffer Data { uvec4 data[]; };

#define MAX_SG 128u
shared float16_t sA[MAX_SG * 32u];
shared float16_t sB[256];  // lK x lN <= 256 (checked by the host)

#define ACC coopmat<ACC_T, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>
#define RM gl_CooperativeMatrixLayoutRowMajor

// Raw data as fp16, exponent bits limited so every value is finite and < 1.
void put(uint o, uvec4 v) {
    f16vec2 h;
    h = unpackFloat2x16(v.x & 0x3BFF3BFFu); sA[o + 0u] = h.x; sA[o + 1u] = h.y;
    h = unpackFloat2x16(v.y & 0x3BFF3BFFu); sA[o + 2u] = h.x; sA[o + 3u] = h.y;
    h = unpackFloat2x16(v.z & 0x3BFF3BFFu); sA[o + 4u] = h.x; sA[o + 5u] = h.y;
    h = unpackFloat2x16(v.w & 0x3BFF3BFFu); sA[o + 6u] = h.x; sA[o + 7u] = h.y;
}

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint tid = gl_GlobalInvocationID.x + pc.dstVecOffset;
    uint T   = pc.numThreads;
    uvec2 acc = uvec2(0u);

    for (uint i = lid; i < lK * lN; i += gl_WorkGroupSize.x)
        sB[i] = float16_t(float(hash32(i ^ pc.keyA.y) & 0xFFu) * (1.0 / 8192.0));
    barrier();
    coopmat<float16_t, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> b;
    coopMatLoad(b, sB, 0, lN, RM);
    coopmat<float16_t, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> a;
    ACC c = ACC(ACC_T(0.0));
    const uint tiles = gl_WorkGroupSize.x * 32u / (lM * lK);

    for (uint it = 0u; it < pc.itersPerThread; it += 4u) {
        uint g0 = (it + 0u) * T + tid;
        uint g1 = (it + 1u) * T + tid;
        uint g2 = (it + 2u) * T + tid;
        uint g3 = (it + 3u) * T + tid;
        uvec4 v0 = data[vecIndex(g0, pc.keyA)];
        uvec4 v1 = data[vecIndex(g1, pc.keyA)];
        uvec4 v2 = data[vecIndex(g2, pc.keyA)];
        uvec4 v3 = data[vecIndex(g3, pc.keyA)];
        acc = mixChecksum(acc, v0, pc.logicalBase + g0);
        acc = mixChecksum(acc, v1, pc.logicalBase + g1);
        acc = mixChecksum(acc, v2, pc.logicalBase + g2);
        acc = mixChecksum(acc, v3, pc.logicalBase + g3);
        uint o = lid * 32u;
        put(o, v0);
        put(o + 8u, v1);
        put(o + 16u, v2);
        put(o + 24u, v3);
        barrier();
        for (uint t = 0u; t < tiles && pc.mode != 0u; ++t) {
            coopMatLoad(a, sA, t * lM * lK, lK, RM);
            for (uint r = 0u; r < pc.mode; ++r) c = coopMatMulAdd(a, b, c);
        }
        barrier();
    }

    publish(acc);
    // Keeps the matrix work alive without touching the verified checksum.
    float s = 0.0;
    for (int i = 0; i < c.length(); ++i) s += float(c[i]);
    if (floatBitsToUint(s) == 0x7FC00001u) atomicAdd(res[pc.outSlot].w, 1u);
}
