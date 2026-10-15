// 1024^3 GEMM on the matrix units (VK_KHR_cooperative_matrix), used only to load the
// GPU before the measurements so DVFS raises the GPU / memory clocks. Not timed.
// C = A x B, A and B fp16 row-major n x n, accumulator ACC_T (fp32 or fp16).
// One subgroup per workgroup computes a 2x2 block of lM x lN tiles.
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x_id = 0) in;            // = subgroup size
layout(constant_id = 1) const uint lM = 16;
layout(constant_id = 2) const uint lN = 16;
layout(constant_id = 3) const uint lK = 16;

layout(push_constant) uniform PC { uint n; } pc;
layout(std430, set = 0, binding = 0) readonly buffer A { float16_t a[]; };
layout(std430, set = 0, binding = 1) readonly buffer B { float16_t b[]; };
layout(std430, set = 0, binding = 2) writeonly buffer C { ACC_T c[]; };

#define ACC coopmat<ACC_T, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>
#define RM gl_CooperativeMatrixLayoutRowMajor

void main() {
    const uint n = pc.n;
    uint tilesN = n / (2u * lN);
    uint row0 = (gl_WorkGroupID.x / tilesN) * 2u * lM;
    uint col0 = (gl_WorkGroupID.x % tilesN) * 2u * lN;

    ACC c00 = ACC(ACC_T(0.0)), c01 = ACC(ACC_T(0.0)), c10 = ACC(ACC_T(0.0)), c11 = ACC(ACC_T(0.0));
    coopmat<float16_t, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> a0, a1;
    coopmat<float16_t, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> b0, b1;
    for (uint k = 0u; k < n; k += lK) {
        coopMatLoad(a0, a, row0 * n + k, n, RM);
        coopMatLoad(a1, a, (row0 + lM) * n + k, n, RM);
        coopMatLoad(b0, b, k * n + col0, n, RM);
        coopMatLoad(b1, b, k * n + col0 + lN, n, RM);
        c00 = coopMatMulAdd(a0, b0, c00);
        c01 = coopMatMulAdd(a0, b1, c01);
        c10 = coopMatMulAdd(a1, b0, c10);
        c11 = coopMatMulAdd(a1, b1, c11);
    }
    coopMatStore(c00, c, row0 * n + col0, n, RM);
    coopMatStore(c01, c, row0 * n + col0 + lN, n, RM);
    coopMatStore(c10, c, (row0 + lM) * n + col0, n, RM);
    coopMatStore(c11, c, (row0 + lM) * n + col0 + lN, n, RM);
}
