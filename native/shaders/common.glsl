// Shared definitions for all benchmark kernels.
//
// IMPORTANT: hash32 / permuteLine / vecIndex / mixChecksum are mirrored bit-exactly
// in native/src/reference.h. The CPU uses them to compute the expected checksum, so any
// change here must be made there as well.

layout(local_size_x_id = 0) in;

layout(push_constant) uniform PC {
    uint chunkWordBase;  // global 32-bit word index of element 0 of the bound chunk
    uint logicalBase;    // added to the logical index g before it is folded into the checksum
    uint linesMask;      // (#lines in the permutation domain) - 1, domain is a power of two
    uint linesBits;      // log2(#lines)
    uint vplShift;       // log2(uvec4 per line); line size = 16 << vplShift bytes
    uint pattern;        // 0 = sequential, 1 = pseudo-random permutation
    uint seedMix;        // hashed data seed (decides the content of memory)
    uint outSlot;        // result slot in the Out buffer
    uvec4 keyA;          // permutation key: read / write / copy-source
    uvec4 keyB;          // permutation key: copy-destination
    uint itersPerThread; // multiple of 4
    uint numThreads;     // total invocations in the dispatch
    uint dstVecOffset;   // copy: destination region offset (uvec4 units) inside the chunk
    uint mode;           // verify: 0 = in-place content, 1 = copy result
} pc;

// Result slot: x,y = checksum (or x = mismatch count), z = completed workgroups.
layout(std430, set = 0, binding = 1) buffer Out { uvec4 res[]; };

uint hash32(uint x) {          // "lowbias32" integer hash
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Content of every 32-bit word of the memory: depends only on its global index and a
// runtime seed, so neither the compiler nor the driver can know it ahead of time.
uint dataValue(uint word) { return hash32(word ^ pc.seedMix); }

// Bijective pseudo-random permutation of [0, 2^linesBits). Each round is invertible
// (xor with key, multiply by odd constant mod 2^n, xorshift-right), so every line is
// visited exactly once per pass -> no reuse, no cache hits, deterministic writes.
uint permuteLine(uint x, uvec4 k, uint pass) {
    if (pc.linesBits == 0u) return 0u;
    uint m  = pc.linesMask;
    uint sh = max((pc.linesBits + 1u) >> 1, 1u);
    x = ((x ^ (k.x ^ (pass * 0x9E3779B9u))) * 0x9E3779B1u) & m; x ^= x >> sh;
    x = ((x ^ k.y) * 0x85EBCA6Bu) & m; x ^= x >> sh;
    x = ((x ^ k.z) * 0xC2B2AE35u) & m; x ^= x >> sh;
    x = ((x ^ k.w) * 0x27D4EB2Fu) & m; x ^= x >> sh;
    return x;
}

// Logical vector index g -> physical uvec4 index. Consecutive g inside one line map to
// consecutive uvec4s, so neighbouring invocations together fetch one whole line
// (16 << vplShift bytes). Line order is permuted; g beyond one pass starts a new pass
// with a different permutation.
uint vecIndex(uint g, uvec4 k) {
    uint lineL = g >> pc.vplShift;
    uint sub   = g & ((1u << pc.vplShift) - 1u);
    uint line  = lineL & pc.linesMask;
    if (pc.pattern != 0u) line = permuteLine(line, k, lineL >> pc.linesBits);
    return (line << pc.vplShift) | sub;
}

uint rotl(uint x, uint r) { return (x << r) | (x >> (32u - r)); }

// Order independent (wrap-around sums), so the result is identical for any scheduling,
// but it depends on both the loaded value and its logical index G: reading the wrong
// address, a stale value or skipping a load changes the checksum.
uvec2 mixChecksum(uvec2 acc, uvec4 v, uint G) {
    acc.x += (v.x ^ G) + rotl(v.y, 7u) + rotl(v.z, 13u) + rotl(v.w, 19u);
    acc.y += (rotl(v.x ^ v.w, 5u) * (G | 1u)) + (v.y ^ v.z);
    return acc;
}

// Fixed size on purpose: an array sized by the gl_WorkGroupSize spec constant is
// mis-allocated by some drivers (Adreno 830: nondeterministic checksums). The host
// rejects --wg > MAX_WG.
#define MAX_WG 1024u
shared uvec2 red[MAX_WG];

// Workgroup tree reduction + one atomic per workgroup (keeps result traffic negligible).
void publish(uvec2 acc) {
    uint lid = gl_LocalInvocationID.x;
    red[lid] = acc;
    barrier();
    for (uint s = gl_WorkGroupSize.x >> 1; s > 0u; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        barrier();
    }
    if (lid == 0u) {
        atomicAdd(res[pc.outSlot].x, red[0].x);
        atomicAdd(res[pc.outSlot].y, red[0].y);
        atomicAdd(res[pc.outSlot].z, 1u);
    }
}
