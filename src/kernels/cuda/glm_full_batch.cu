#include "strata/kernels/glm_full_batch.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::full_glm_dsa {
namespace {
constexpr int NT = 256;
void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "glm_full_batch: %s\n", what); std::abort(); }
}
void check(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }

// One block owns a row, including the residual update. Two strided loops avoid
// the fixed v[16] per-thread array that limits Flash's rows to 4096 elements.
// The reduction barrier separates all residual writes from in-place RMS writes.
__global__ void residual_norm_rows_kernel(float* h, const float* add, const float* w,
                                          float eps, int E, float* x, uint16_t* x16) {
    const size_t base = (size_t) blockIdx.x*E;
    const int tid = threadIdx.x;
    __shared__ float partial[NT];
    float ss = 0;
    for (int e = tid; e < E; e += NT) {
        const float v = h[base+e] + (add ? add[base+e] : 0);
        h[base+e] = v;
        ss += v*v;
    }
    partial[tid] = ss;
    __syncthreads();
    for (int d = NT/2; d > 0; d >>= 1) {
        if (tid < d) partial[tid] += partial[tid+d];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0]/E + eps);
    for (int e = tid; e < E; e += NT) {
        const float v = h[base+e]*inv*(w ? w[e] : 1);
        x[base+e] = v;
        if (x16) x16[base+e] = f16_from_f32(v);
    }
}
}  // namespace

void residual_norm_rows(float* h, const float* add, const float* w, float eps,
                        int T, int E, float* x, void* x16, void* stream) {
    if (T <= 0) return;
    require(E > 0 && eps >= 0, "invalid RMS shape/epsilon");
    residual_norm_rows_kernel<<<T, NT, 0, (cudaStream_t) stream>>>(h, add, w, eps, E, x, (uint16_t*) x16);
    check(cudaGetLastError());
}
void add_residual_rows(float* h, const float* add, int T, int E, void* stream) {
    require(T >= 0 && E > 0 && (int64_t) T*E <= INT_MAX, "invalid residual shape");
    add_residual(h, add, T*E, stream);
}
void prepare_rows(float* q, const float* kv, const float* norm, uint16_t* cache,
                  int p0, int T, int heads, int nope, int latent, int rope,
                  float eps, Rope rotary, void* stream) {
    require(p0 >= 0 && T >= 0 && (int64_t) p0+T <= INT_MAX, "invalid cache positions");
    for (int t = 0; t < T; ++t)
        prepare(q+(size_t) t*heads*(nope+rope), kv+(size_t) t*(latent+rope), norm,
                cache, p0+t, heads, nope, latent, rope, eps, rotary, stream);
}
void index_rows(float* iq, float* ik, const float* w, const float* b, float* cache,
                const float* iw, int p0, int T, int heads, int dim, int rope,
                int topk, float eps, Rope rotary, float* scratch, int* cells, void* stream) {
    require(p0 >= 0 && T >= 0 && (int64_t) p0+T <= INT_MAX && topk > 0 && topk <= 2048,
            "invalid indexer positions/top-k");
    if (T == 0) return;
    check(cudaMemsetAsync(cells, 0xff, (size_t) T*topk*sizeof(int), (cudaStream_t) stream));
    // Populate all new keys first. Causality belongs to each score call's cache
    // length, so later keys already in memory cannot leak into an earlier query.
    for (int t = 0; t < T; ++t)
        index_prepare(iq+(size_t) t*heads*dim, ik+(size_t) t*dim, w, b, cache,
                      p0+t, heads, dim, rope, eps, rotary, stream);
    for (int t = 0; t < T; ++t) {
        score(iq+(size_t) t*heads*dim, cache, iw+(size_t) t*heads,
              dim, heads, p0+t+1, scratch, stream);
        select(scratch, p0+t+1, topk, cells+(size_t) t*topk, stream);
    }
}
// mla_rows lives beside the shared single-token MLA kernel in glm_full_dsa.cu.
}  // namespace strata::kernels::full_glm_dsa
