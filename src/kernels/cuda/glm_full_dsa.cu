// Full GLM-5.3 DSA. Math follows ggml-org/llama.cpp src/models/glm-dsa.cpp
// (retrieved 2026-10-16) and zai-org/GLM-5.3 config.json. No Flash k-pooling.
// Stable radix top-k adapts this repository's dsa_topk.cuh to 2048 cells.
#include "strata/kernels/glm_full_dsa.hpp"
#include "strata/kernels/glm_full_batch.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::full_glm_dsa {
namespace {
constexpr int NT = 256;

void require(bool yes, const char* what) {
    if (!yes) { std::fprintf(stderr, "full_glm_dsa: %s\n", what); std::abort(); }
}
void check() {
    const auto e = cudaGetLastError();
    require(e == cudaSuccess, cudaGetErrorString(e));
}
// A block reduction using shared memory rather than CUDA warp-width assumptions.
__device__ float sum(float x, float* tmp) {
    const int t = threadIdx.x;
    tmp[t] = x;
    __syncthreads();
    for (int d = NT / 2; d; d >>= 1) {
        if (t < d) tmp[t] += tmp[t + d];
        __syncthreads();
    }
    const float out = tmp[0];
    __syncthreads();
    return out;
}
__device__ void rotate(float* row, int n, int pos, Rope r) {
    for (int i = 2 * threadIdx.x; i < n; i += 2 * blockDim.x) {
        const float angle = pos * powf(r.base, -(float) i / n) * r.freq_scale;
        const float c = cosf(angle) * r.attn_factor, s = sinf(angle) * r.attn_factor;
        const float a = row[i], b = row[i + 1];
        row[i] = a*c - b*s;
        row[i + 1] = a*s + b*c;
    }
}
__global__ void norm_kernel(float* h, const float* add, const float* w, float eps, int n, float* x) {
    __shared__ float tmp[NT];
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += NT) {
        const float v = h[i] + (add ? add[i] : 0);
        h[i] = v;
        ss += v*v;
    }
    const float inv = rsqrtf(sum(ss, tmp) / n + eps);
    for (int i = threadIdx.x; i < n; i += NT) x[i] = h[i] * inv * (w ? w[i] : 1);
}
__global__ void add_kernel(float* h, const float* add, int n) {
    const int i = blockIdx.x * NT + threadIdx.x;
    if (i < n) h[i] += add[i];
}
__global__ void prep_kernel(float* q, const float* kv, const float* norm, uint16_t* cache,
                            int pos, int heads, int nope, int latent, int rope, float eps, Rope r) {
    if (blockIdx.x < heads) {
        rotate(q + (size_t) blockIdx.x * (nope + rope) + nope, rope, pos, r);
        return;
    }
    __shared__ float tmp[NT];
    __shared__ float pe[1024];
    float ss = 0;
    for (int i = threadIdx.x; i < latent; i += NT) ss += kv[i]*kv[i];
    const float inv = rsqrtf(sum(ss, tmp) / latent + eps);
    uint16_t* row = cache + (size_t) pos * (latent + rope);
    for (int i = threadIdx.x; i < latent; i += NT) row[i] = f16_from_f32(kv[i]*inv*norm[i]);
    for (int i = threadIdx.x; i < rope; i += NT) pe[i] = kv[latent + i];
    __syncthreads();
    rotate(pe, rope, pos, r);
    __syncthreads();
    for (int i = threadIdx.x; i < rope; i += NT) row[latent + i] = f16_from_f32(pe[i]);
}
__global__ void index_prep_kernel(float* iq, float* ik, const float* w, const float* b, float* cache,
                                  int pos, int heads, int dim, int rope, float eps, Rope r) {
    __shared__ float x[1024], tmp[NT];
    const bool key = blockIdx.x == heads;
    float* row = key ? ik : iq + (size_t) blockIdx.x * dim;
    float sm = 0;
    for (int i = threadIdx.x; i < dim; i += NT) sm += row[i];
    const float mean = sum(sm, tmp) / dim;
    float ss = 0;
    for (int i = threadIdx.x; i < dim; i += NT) { const float v = row[i] - mean; ss += v*v; }
    const float inv = rsqrtf(sum(ss, tmp) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += NT)
        x[i] = key ? (row[i] - mean)*inv*w[i] + b[i] : row[i];
    __syncthreads();
    rotate(x, rope, pos, r);
    __syncthreads();
    // Sylvester H: [H H; H -H], exactly the ggml KV-cache rotation matrix.
    // Read the whole butterfly before writes; register slots handle dim>NT.
    for (int d = 1; d < dim; d <<= 1) {
        float next[4];
        for (int i = threadIdx.x, k = 0; i < dim; i += NT, ++k)
            next[k] = (i & d) ? x[i ^ d] - x[i] : x[i] + x[i ^ d];
        __syncthreads();
        for (int i = threadIdx.x, k = 0; i < dim; i += NT, ++k) x[i] = next[k];
        __syncthreads();
    }
    for (int i = threadIdx.x; i < dim; i += NT) {
        const float v = x[i] * rsqrtf((float) dim);
        row[i] = v;
        if (key) cache[(size_t) pos * dim + i] = v;
    }
}
__global__ void score_kernel(const float* iq, const float* cache, const float* iw,
                             int dim, int heads, int ncache, float* out) {
    const int cell = blockIdx.x * NT + threadIdx.x;
    if (cell >= ncache) return;
    float value = 0;
    for (int h = 0; h < heads; ++h) {
        float dot = 0;
        for (int i = 0; i < dim; ++i) dot += iq[(size_t) h*dim+i]*cache[(size_t) cell*dim+i];
        value += fmaxf(dot, 0)*iw[h];
    }
    out[cell] = value * rsqrtf((float) dim*heads);
}
__device__ unsigned key(float f) {
    const unsigned u = __float_as_uint(f == 0 ? 0 : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ void select_kernel(const float* scores, int ncache, int top, int* cells) {
    __shared__ unsigned hist[256], prefix, mask;
    __shared__ int need, gt[NT], eq[NT];
    __shared__ unsigned long long winners[2048];
    const int t = threadIdx.x;
    if (t == 0) { prefix = mask = 0; need = top; }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0;
        __syncthreads();
        for (int i = t; i < ncache; i += NT) {
            const unsigned k = key(scores[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t == 0) {
            for (int b = 255; b >= 0; --b) {
                if ((int) hist[b] >= need) {
                    prefix |= (unsigned) b << shift;
                    mask |= 255u << shift;
                    break;
                }
                need -= hist[b];
            }
        }
        __syncthreads();
    }
    const int chunk = (ncache + NT - 1) / NT, lo = min(ncache, t*chunk), hi = min(ncache, lo+chunk);
    int ng = 0, ne = 0;
    for (int i = lo; i < hi; ++i) { ng += key(scores[i]) > prefix; ne += key(scores[i]) == prefix; }
    gt[t] = ng; eq[t] = ne;
    __syncthreads();
    for (int off = 1; off < NT; off <<= 1) {
        const int g = t >= off ? gt[t-off] : 0, e = t >= off ? eq[t-off] : 0;
        __syncthreads();
        gt[t] += g; eq[t] += e;
        __syncthreads();
    }
    int og = gt[t]-ng, oe = eq[t]-ne;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = key(scores[i]);
        const auto packed = ((unsigned long long) k << 32) | (unsigned) ~i;
        if (k > prefix) winners[og++] = packed;
        else if (k == prefix) { if (oe < need) winners[gt[NT-1]+oe] = packed; ++oe; }
    }
    int n2 = 1;
    while (n2 < top) n2 <<= 1;
    for (int i = top+t; i < n2; i += NT) winners[i] = 0;
    __syncthreads();
    for (int size = 2; size <= n2; size <<= 1)
        for (int stride = size >> 1; stride; stride >>= 1) {
            for (int i = t; i < n2; i += NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const auto a = winners[i], b = winners[j];
                    if ((i & size) == 0 ? a < b : a > b) { winners[i] = b; winners[j] = a; }
                }
            }
            __syncthreads();
        }
    for (int i = t; i < top; i += NT) cells[i] = (int) ~(unsigned) winners[i];
}
__device__ float bf16(uint16_t x) { return __uint_as_float((unsigned) x << 16); }
// FP16 -> F32 is exact, including subnormals. Use the hardware conversion already
// used by Flash's latent reader; keep software FP16 rounding for cache writes.
// This changes neither stored bits nor the order of any attention arithmetic.
__device__ __forceinline__ float latent_f32(uint16_t x) { return __half2float(__ushort_as_half(x)); }
template<bool Rows>
__global__ void mla_kernel(const float* q, const uint16_t* kb, const uint16_t* vb,
                           const uint16_t* cache, const int* cells, int nsel, int nope,
                           int latent, int rope, int vhead, float scale, float* out,
                           int heads, int p0) {
    const int h = Rows ? blockIdx.x % heads : blockIdx.x, t = threadIdx.x;
    if constexpr (Rows) {
        // Flatten (row,head) so long batches do not inherit CUDA grid.y's limit.
        // Every block retains the token kernel's exact head-local arithmetic.
        const int row = blockIdx.x / heads;
        q += (size_t) row*heads*(nope+rope);
        cells += (size_t) row*nsel; // nsel is the padded row stride until this point
        out += (size_t) row*heads*vhead;
        nsel = min(nsel, p0+row+1);
    }
    extern __shared__ float buf[];
    float* probs = buf, *qa = buf+nsel, *ctx = qa+latent;
    const float* query = q + (size_t) h*(nope+rope);
    for (int c = t; c < latent; c += NT) {
        float dot = 0;
        for (int i = 0; i < nope; ++i) dot += bf16(kb[((size_t) h*latent+c)*nope+i])*query[i];
        qa[c] = dot;
    }
    __syncthreads();
    for (int s = t; s < nsel; s += NT) {
        const int cell = cells[s];
        if (cell < 0) { probs[s] = -INFINITY; continue; }
        const uint16_t* row = cache + (size_t) cell*(latent+rope);
        float dot = 0;
        for (int c = 0; c < latent; ++c) dot += qa[c]*latent_f32(row[c]);
        for (int c = 0; c < rope; ++c) dot += query[nope+c]*latent_f32(row[latent+c]);
        probs[s] = dot*scale;
    }
    __syncthreads();
    if constexpr (Rows) {
        // Row grids already fill the GPU. The original serial softmax avoids
        // extra block barriers and measures faster in that throughput regime.
        if (t == 0) {
            float mx = -INFINITY, denom = 0;
            for (int s = 0; s < nsel; ++s) mx = fmaxf(mx, probs[s]);
            for (int s = 0; s < nsel; ++s) {
                probs[s] = probs[s] == -INFINITY ? 0 : expf(probs[s]-mx);
                denom += probs[s];
            }
            for (int s = 0; s < nsel; ++s) probs[s] = denom > 0 ? probs[s]/denom : 0;
        }
    } else {
        __shared__ float softmax_max, softmax_denom;
        if (t == 0) {
            float mx = -INFINITY;
            for (int s = 0; s < nsel; ++s) mx = fmaxf(mx, probs[s]);
            softmax_max = mx;
        }
        __syncthreads();
        // These elementwise operations are independent. Keep the original expf
        // and division, and the denominator's left-to-right F32 sum on one thread;
        // parallel work changes scheduling, not any value's arithmetic order.
        for (int s = t; s < nsel; s += NT)
            probs[s] = probs[s] == -INFINITY ? 0 : expf(probs[s]-softmax_max);
        __syncthreads();
        if (t == 0) {
            float denom = 0;
            for (int s = 0; s < nsel; ++s) denom += probs[s];
            softmax_denom = denom;
        }
        __syncthreads();
        for (int s = t; s < nsel; s += NT)
            probs[s] = softmax_denom > 0 ? probs[s]/softmax_denom : 0;
    }
    __syncthreads();
    for (int c = t; c < latent; c += NT) {
        float value = 0;
        for (int s = 0; s < nsel; ++s)
            if (cells[s] >= 0) value += probs[s]*latent_f32(cache[(size_t) cells[s]*(latent+rope)+c]);
        ctx[c] = value;
    }
    __syncthreads();
    for (int v = t; v < vhead; v += NT) {
        float value = 0;
        for (int c = 0; c < latent; ++c) value += bf16(vb[((size_t) h*vhead+v)*latent+c])*ctx[c];
        out[(size_t) h*vhead+v] = value;
    }
}
void check_rope(int rope, Rope r) {
    require(rope > 0 && rope <= 1024 && rope % 2 == 0 && r.base > 0 && r.freq_scale > 0,
            "invalid RoPE dimensions/frequencies");
}
}  // namespace

void residual_norm(float* h, const float* add, const float* w, float eps, int n,
                   float* x, void* xq, void* stream) {
    require(n > 0 && eps >= 0 && (!xq || n % 32 == 0), "invalid norm shape/epsilon");
    norm_kernel<<<1, NT, 0, (cudaStream_t) stream>>>(h, add, w, eps, n, x);
    check();
    if (xq) glmf::quantize_q8_1(x, xq, n, (cudaStream_t) stream);
}
void add_residual(float* h, const float* add, int n, void* stream) {
    if (n <= 0) return;
    add_kernel<<<(n+NT-1)/NT, NT, 0, (cudaStream_t) stream>>>(h, add, n);
    check();
}
void prepare(float* q, const float* kv, const float* norm, uint16_t* cache, int pos,
             int heads, int nope, int latent, int rope, float eps, Rope r, void* stream) {
    check_rope(rope, r);
    require(pos >= 0 && heads > 0 && nope > 0 && latent > 0 && eps >= 0, "invalid prepare shape");
    prep_kernel<<<heads+1, NT, 0, (cudaStream_t) stream>>>(q, kv, norm, cache, pos, heads, nope, latent, rope, eps, r);
    check();
}
void index_prepare(float* iq, float* ik, const float* w, const float* b, float* cache,
                   int pos, int heads, int dim, int rope, float eps, Rope r, void* stream) {
    check_rope(rope, r);
    require(pos >= 0 && heads > 0 && dim >= rope && dim <= 1024 && (dim & (dim-1)) == 0 && eps >= 0,
            "invalid indexer shape");
    index_prep_kernel<<<heads+1, NT, 0, (cudaStream_t) stream>>>(iq, ik, w, b, cache, pos, heads, dim, rope, eps, r);
    check();
}
void score(const float* iq, const float* cache, const float* iw, int dim, int heads,
           int ncache, float* out, void* stream) {
    require(dim > 0 && heads > 0 && ncache > 0, "invalid score shape");
    score_kernel<<<(ncache+NT-1)/NT, NT, 0, (cudaStream_t) stream>>>(iq, cache, iw, dim, heads, ncache, out);
    check();
}
void select(const float* scores, int ncache, int topk, int* cells, void* stream) {
    require(ncache > 0 && topk > 0 && topk <= 2048, "invalid top-k shape");
    select_kernel<<<1, NT, 0, (cudaStream_t) stream>>>(scores, ncache, ncache < topk ? ncache : topk, cells);
    check();
}
void mla(const float* q, const uint16_t* kb, const uint16_t* vb, const uint16_t* cache,
         const int* cells, int nsel, int heads, int nope, int latent, int rope, int vhead,
         float scale, float* out, void* stream) {
    require(nsel > 0 && nsel <= 2048 && heads > 0 && nope > 0 && latent > 0 && vhead > 0,
            "invalid MLA shape");
    const size_t bytes = (size_t) (nsel+2*latent)*sizeof(float);
    mla_kernel<false><<<heads, NT, bytes, (cudaStream_t) stream>>>(q, kb, vb, cache, cells, nsel, nope,
        latent, rope, vhead, scale, out, heads, 0);
    check();
}
void mla_rows(const float* q, const uint16_t* kb, const uint16_t* vb,
              const uint16_t* cache, const int* cells, int p0, int T, int topk,
              int heads, int nope, int latent, int rope, int vhead, float scale,
              float* out, void* stream) {
    require(p0 >= 0 && T >= 0 && (int64_t) p0+T <= INT_MAX && topk > 0 && topk <= 2048 &&
            heads > 0 && (int64_t) T*heads <= INT_MAX && nope > 0 && latent > 0 && vhead > 0,
            "invalid batched MLA shape");
    if (T == 0) return;
    const size_t bytes = (size_t) (topk+2*latent)*sizeof(float);
    mla_kernel<true><<<T*heads, NT, bytes, (cudaStream_t) stream>>>(q, kb, vb, cache, cells, topk, nope,
        latent, rope, vhead, scale, out, heads, p0);
    check();
}
}  // namespace strata::kernels::full_glm_dsa
